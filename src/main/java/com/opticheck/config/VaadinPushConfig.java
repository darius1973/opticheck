package com.opticheck.config;

import com.vaadin.flow.component.page.AppShellConfigurator;
import com.vaadin.flow.component.page.Push;
import org.springframework.stereotype.Component;

@Component
@Push
public class VaadinPushConfig implements AppShellConfigurator {
    // No methods needed. The presence of @Push enables server push.
}

