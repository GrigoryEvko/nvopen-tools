// Function: sub_E058A0
// Address: 0xe058a0
//
const char *__fastcall sub_E058A0(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 0x2305 )
  {
    if ( a1 > 0x3E0C )
    {
      switch ( a1 )
      {
        case 0x3FE1u:
          result = "DW_AT_APPLE_optimized";
          break;
        case 0x3FE2u:
          result = "DW_AT_APPLE_flags";
          break;
        case 0x3FE3u:
          result = "DW_AT_APPLE_isa";
          break;
        case 0x3FE4u:
          result = "DW_AT_APPLE_block";
          break;
        case 0x3FE5u:
          result = "DW_AT_APPLE_major_runtime_vers";
          break;
        case 0x3FE6u:
          result = "DW_AT_APPLE_runtime_class";
          break;
        case 0x3FE7u:
          result = "DW_AT_APPLE_omit_frame_ptr";
          break;
        case 0x3FE8u:
          result = "DW_AT_APPLE_property_name";
          break;
        case 0x3FE9u:
          result = "DW_AT_APPLE_property_getter";
          break;
        case 0x3FEAu:
          result = "DW_AT_APPLE_property_setter";
          break;
        case 0x3FEBu:
          result = "DW_AT_APPLE_property_attribute";
          break;
        case 0x3FECu:
          result = "DW_AT_APPLE_objc_complete_type";
          break;
        case 0x3FEDu:
          result = "DW_AT_APPLE_property";
          break;
        case 0x3FEEu:
          result = "DW_AT_APPLE_objc_direct";
          break;
        case 0x3FEFu:
          result = "DW_AT_APPLE_sdk";
          break;
        case 0x3FF0u:
          result = "DW_AT_APPLE_origin";
          break;
        case 0x3FF1u:
          result = "DW_AT_APPLE_enum_kind";
          break;
        default:
          return 0;
      }
    }
    else if ( a1 > 0x3DFF )
    {
      switch ( a1 )
      {
        case 0x3E01u:
          result = "DW_AT_LLVM_config_macros";
          break;
        case 0x3E02u:
          result = "DW_AT_LLVM_sysroot";
          break;
        case 0x3E03u:
          result = "DW_AT_LLVM_tag_offset";
          break;
        case 0x3E04u:
          result = "DW_AT_LLVM_ptrauth_key";
          break;
        case 0x3E05u:
          result = "DW_AT_LLVM_ptrauth_address_discriminated";
          break;
        case 0x3E06u:
          result = "DW_AT_LLVM_ptrauth_extra_discriminator";
          break;
        case 0x3E07u:
          result = "DW_AT_LLVM_apinotes";
          break;
        case 0x3E08u:
          result = "DW_AT_LLVM_ptrauth_isa_pointer";
          break;
        case 0x3E09u:
          result = "DW_AT_LLVM_ptrauth_authenticates_null_values";
          break;
        case 0x3E0Au:
          result = "DW_AT_LLVM_ptrauth_authentication_mode";
          break;
        case 0x3E0Bu:
          result = "DW_AT_LLVM_num_extra_inhabitants";
          break;
        case 0x3E0Cu:
          result = "DW_AT_LLVM_stmt_sequence";
          break;
        default:
          result = "DW_AT_LLVM_include_path";
          break;
      }
    }
    else
    {
      if ( a1 > 0x3B31 )
        return 0;
      if ( a1 <= 0x3B10 )
      {
        if ( a1 == 12816 )
          return "DW_AT_UPC_threads_scaled";
        if ( a1 > 0x3210 )
        {
          if ( a1 == 14848 )
            return "DW_AT_PGI_lbase";
          if ( a1 <= 0x3A00 )
          {
            result = "DW_AT_IBM_home_location";
            switch ( a1 )
            {
              case 0x393Fu:
                return result;
              case 0x3940u:
                return "DW_AT_IBM_alt_srcview";
              case 0x393Eu:
                return "DW_AT_IBM_wsa_addr";
            }
          }
          else
          {
            result = "DW_AT_PGI_soffset";
            if ( a1 == 14849 )
              return result;
            if ( a1 == 14850 )
              return "DW_AT_PGI_lstride";
          }
          return 0;
        }
        if ( a1 == 9987 )
        {
          return "DW_AT_NV_general_flags";
        }
        else
        {
          switch ( a1 )
          {
            case 0x2900u:
              result = "DW_AT_GO_kind";
              break;
            case 0x2901u:
              result = "DW_AT_GO_key";
              break;
            case 0x2902u:
              result = "DW_AT_GO_elem";
              break;
            case 0x2903u:
              result = "DW_AT_GO_embedded_field";
              break;
            case 0x2904u:
              result = "DW_AT_GO_runtime_type";
              break;
            default:
              return 0;
          }
        }
      }
      else
      {
        switch ( a1 )
        {
          case 0x3B11u:
            result = "DW_AT_BORLAND_property_read";
            break;
          case 0x3B12u:
            result = "DW_AT_BORLAND_property_write";
            break;
          case 0x3B13u:
            result = "DW_AT_BORLAND_property_implements";
            break;
          case 0x3B14u:
            result = "DW_AT_BORLAND_property_index";
            break;
          case 0x3B15u:
            result = "DW_AT_BORLAND_property_default";
            break;
          case 0x3B20u:
            result = "DW_AT_BORLAND_Delphi_unit";
            break;
          case 0x3B21u:
            result = "DW_AT_BORLAND_Delphi_class";
            break;
          case 0x3B22u:
            result = "DW_AT_BORLAND_Delphi_record";
            break;
          case 0x3B23u:
            result = "DW_AT_BORLAND_Delphi_metaclass";
            break;
          case 0x3B24u:
            result = "DW_AT_BORLAND_Delphi_constructor";
            break;
          case 0x3B25u:
            result = "DW_AT_BORLAND_Delphi_destructor";
            break;
          case 0x3B26u:
            result = "DW_AT_BORLAND_Delphi_anonymous_method";
            break;
          case 0x3B27u:
            result = "DW_AT_BORLAND_Delphi_interface";
            break;
          case 0x3B28u:
            result = "DW_AT_BORLAND_Delphi_ABI";
            break;
          case 0x3B29u:
            result = "DW_AT_BORLAND_Delphi_return";
            break;
          case 0x3B30u:
            result = "DW_AT_BORLAND_Delphi_frameptr";
            break;
          case 0x3B31u:
            result = "DW_AT_BORLAND_closure";
            break;
          default:
            return 0;
        }
      }
    }
  }
  else
  {
    if ( a1 <= 0x2000 )
    {
      if ( a1 > 0x8C )
      {
        result = "DW_AT_GHS_using_namespace";
        switch ( a1 )
        {
          case 0x807u:
            return result;
          case 0x808u:
            return "DW_AT_GHS_using_declaration";
          case 0x806u:
            return "DW_AT_GHS_namespace_alias";
        }
      }
      else if ( a1 )
      {
        switch ( a1 )
        {
          case 1u:
            result = "DW_AT_sibling";
            break;
          case 2u:
            result = "DW_AT_location";
            break;
          case 3u:
            result = "DW_AT_name";
            break;
          case 9u:
            result = "DW_AT_ordering";
            break;
          case 0xBu:
            result = "DW_AT_byte_size";
            break;
          case 0xCu:
            result = "DW_AT_bit_offset";
            break;
          case 0xDu:
            result = "DW_AT_bit_size";
            break;
          case 0x10u:
            result = "DW_AT_stmt_list";
            break;
          case 0x11u:
            result = "DW_AT_low_pc";
            break;
          case 0x12u:
            result = "DW_AT_high_pc";
            break;
          case 0x13u:
            result = "DW_AT_language";
            break;
          case 0x15u:
            result = "DW_AT_discr";
            break;
          case 0x16u:
            result = "DW_AT_discr_value";
            break;
          case 0x17u:
            result = "DW_AT_visibility";
            break;
          case 0x18u:
            result = "DW_AT_import";
            break;
          case 0x19u:
            result = "DW_AT_string_length";
            break;
          case 0x1Au:
            result = "DW_AT_common_reference";
            break;
          case 0x1Bu:
            result = "DW_AT_comp_dir";
            break;
          case 0x1Cu:
            result = "DW_AT_const_value";
            break;
          case 0x1Du:
            result = "DW_AT_containing_type";
            break;
          case 0x1Eu:
            result = "DW_AT_default_value";
            break;
          case 0x20u:
            result = "DW_AT_inline";
            break;
          case 0x21u:
            result = "DW_AT_is_optional";
            break;
          case 0x22u:
            result = "DW_AT_lower_bound";
            break;
          case 0x25u:
            result = "DW_AT_producer";
            break;
          case 0x27u:
            result = "DW_AT_prototyped";
            break;
          case 0x2Au:
            result = "DW_AT_return_addr";
            break;
          case 0x2Cu:
            result = "DW_AT_start_scope";
            break;
          case 0x2Eu:
            result = "DW_AT_bit_stride";
            break;
          case 0x2Fu:
            result = "DW_AT_upper_bound";
            break;
          case 0x31u:
            result = "DW_AT_abstract_origin";
            break;
          case 0x32u:
            result = "DW_AT_accessibility";
            break;
          case 0x33u:
            result = "DW_AT_address_class";
            break;
          case 0x34u:
            result = "DW_AT_artificial";
            break;
          case 0x35u:
            result = "DW_AT_base_types";
            break;
          case 0x36u:
            result = "DW_AT_calling_convention";
            break;
          case 0x37u:
            result = "DW_AT_count";
            break;
          case 0x38u:
            result = "DW_AT_data_member_location";
            break;
          case 0x39u:
            result = "DW_AT_decl_column";
            break;
          case 0x3Au:
            result = "DW_AT_decl_file";
            break;
          case 0x3Bu:
            result = "DW_AT_decl_line";
            break;
          case 0x3Cu:
            result = "DW_AT_declaration";
            break;
          case 0x3Du:
            result = "DW_AT_discr_list";
            break;
          case 0x3Eu:
            result = "DW_AT_encoding";
            break;
          case 0x3Fu:
            result = "DW_AT_external";
            break;
          case 0x40u:
            result = "DW_AT_frame_base";
            break;
          case 0x41u:
            result = "DW_AT_friend";
            break;
          case 0x42u:
            result = "DW_AT_identifier_case";
            break;
          case 0x43u:
            result = "DW_AT_macro_info";
            break;
          case 0x44u:
            result = "DW_AT_namelist_item";
            break;
          case 0x45u:
            result = "DW_AT_priority";
            break;
          case 0x46u:
            result = "DW_AT_segment";
            break;
          case 0x47u:
            result = "DW_AT_specification";
            break;
          case 0x48u:
            result = "DW_AT_static_link";
            break;
          case 0x49u:
            result = "DW_AT_type";
            break;
          case 0x4Au:
            result = "DW_AT_use_location";
            break;
          case 0x4Bu:
            result = "DW_AT_variable_parameter";
            break;
          case 0x4Cu:
            result = "DW_AT_virtuality";
            break;
          case 0x4Du:
            result = "DW_AT_vtable_elem_location";
            break;
          case 0x4Eu:
            result = "DW_AT_allocated";
            break;
          case 0x4Fu:
            result = "DW_AT_associated";
            break;
          case 0x50u:
            result = "DW_AT_data_location";
            break;
          case 0x51u:
            result = "DW_AT_byte_stride";
            break;
          case 0x52u:
            result = "DW_AT_entry_pc";
            break;
          case 0x53u:
            result = "DW_AT_use_UTF8";
            break;
          case 0x54u:
            result = "DW_AT_extension";
            break;
          case 0x55u:
            result = "DW_AT_ranges";
            break;
          case 0x56u:
            result = "DW_AT_trampoline";
            break;
          case 0x57u:
            result = "DW_AT_call_column";
            break;
          case 0x58u:
            result = "DW_AT_call_file";
            break;
          case 0x59u:
            result = "DW_AT_call_line";
            break;
          case 0x5Au:
            result = "DW_AT_description";
            break;
          case 0x5Bu:
            result = "DW_AT_binary_scale";
            break;
          case 0x5Cu:
            result = "DW_AT_decimal_scale";
            break;
          case 0x5Du:
            result = "DW_AT_small";
            break;
          case 0x5Eu:
            result = "DW_AT_decimal_sign";
            break;
          case 0x5Fu:
            result = "DW_AT_digit_count";
            break;
          case 0x60u:
            result = "DW_AT_picture_string";
            break;
          case 0x61u:
            result = "DW_AT_mutable";
            break;
          case 0x62u:
            result = "DW_AT_threads_scaled";
            break;
          case 0x63u:
            result = "DW_AT_explicit";
            break;
          case 0x64u:
            result = "DW_AT_object_pointer";
            break;
          case 0x65u:
            result = "DW_AT_endianity";
            break;
          case 0x66u:
            result = "DW_AT_elemental";
            break;
          case 0x67u:
            result = "DW_AT_pure";
            break;
          case 0x68u:
            result = "DW_AT_recursive";
            break;
          case 0x69u:
            result = "DW_AT_signature";
            break;
          case 0x6Au:
            result = "DW_AT_main_subprogram";
            break;
          case 0x6Bu:
            result = "DW_AT_data_bit_offset";
            break;
          case 0x6Cu:
            result = "DW_AT_const_expr";
            break;
          case 0x6Du:
            result = "DW_AT_enum_class";
            break;
          case 0x6Eu:
            result = "DW_AT_linkage_name";
            break;
          case 0x6Fu:
            result = "DW_AT_string_length_bit_size";
            break;
          case 0x70u:
            result = "DW_AT_string_length_byte_size";
            break;
          case 0x71u:
            result = "DW_AT_rank";
            break;
          case 0x72u:
            result = "DW_AT_str_offsets_base";
            break;
          case 0x73u:
            result = "DW_AT_addr_base";
            break;
          case 0x74u:
            result = "DW_AT_rnglists_base";
            break;
          case 0x75u:
            result = "DW_AT_dwo_id";
            break;
          case 0x76u:
            result = "DW_AT_dwo_name";
            break;
          case 0x77u:
            result = "DW_AT_reference";
            break;
          case 0x78u:
            result = "DW_AT_rvalue_reference";
            break;
          case 0x79u:
            result = "DW_AT_macros";
            break;
          case 0x7Au:
            result = "DW_AT_call_all_calls";
            break;
          case 0x7Bu:
            result = "DW_AT_call_all_source_calls";
            break;
          case 0x7Cu:
            result = "DW_AT_call_all_tail_calls";
            break;
          case 0x7Du:
            result = "DW_AT_call_return_pc";
            break;
          case 0x7Eu:
            result = "DW_AT_call_value";
            break;
          case 0x7Fu:
            result = "DW_AT_call_origin";
            break;
          case 0x80u:
            result = "DW_AT_call_parameter";
            break;
          case 0x81u:
            result = "DW_AT_call_pc";
            break;
          case 0x82u:
            result = "DW_AT_call_tail_call";
            break;
          case 0x83u:
            result = "DW_AT_call_target";
            break;
          case 0x84u:
            result = "DW_AT_call_target_clobbered";
            break;
          case 0x85u:
            result = "DW_AT_call_data_location";
            break;
          case 0x86u:
            result = "DW_AT_call_data_value";
            break;
          case 0x87u:
            result = "DW_AT_noreturn";
            break;
          case 0x88u:
            result = "DW_AT_alignment";
            break;
          case 0x89u:
            result = "DW_AT_export_symbols";
            break;
          case 0x8Au:
            result = "DW_AT_deleted";
            break;
          case 0x8Bu:
            result = "DW_AT_defaulted";
            break;
          case 0x8Cu:
            result = "DW_AT_loclists_base";
            break;
          default:
            return 0;
        }
        return result;
      }
      return 0;
    }
    switch ( a1 )
    {
      case 0x2001u:
        result = "DW_AT_MIPS_fde";
        break;
      case 0x2002u:
        result = "DW_AT_MIPS_loop_begin";
        break;
      case 0x2003u:
        result = "DW_AT_MIPS_tail_loop_begin";
        break;
      case 0x2004u:
        result = "DW_AT_MIPS_epilog_begin";
        break;
      case 0x2005u:
        result = "DW_AT_MIPS_loop_unroll_factor";
        break;
      case 0x2006u:
        result = "DW_AT_MIPS_software_pipeline_depth";
        break;
      case 0x2007u:
        result = "DW_AT_MIPS_linkage_name";
        break;
      case 0x2008u:
        result = "DW_AT_MIPS_stride";
        break;
      case 0x2009u:
        result = "DW_AT_MIPS_abstract_name";
        break;
      case 0x200Au:
        result = "DW_AT_MIPS_clone_origin";
        break;
      case 0x200Bu:
        result = "DW_AT_MIPS_has_inlines";
        break;
      case 0x200Cu:
        result = "DW_AT_MIPS_stride_byte";
        break;
      case 0x200Du:
        result = "DW_AT_MIPS_stride_elem";
        break;
      case 0x200Eu:
        result = "DW_AT_MIPS_ptr_dopetype";
        break;
      case 0x200Fu:
        result = "DW_AT_MIPS_allocatable_dopetype";
        break;
      case 0x2010u:
        result = "DW_AT_MIPS_assumed_shape_dopetype";
        break;
      case 0x2011u:
        result = "DW_AT_MIPS_assumed_size";
        break;
      case 0x2012u:
        result = "DW_AT_HP_raw_data_ptr";
        break;
      case 0x2013u:
        result = "DW_AT_HP_pass_by_reference";
        break;
      case 0x2014u:
        result = "DW_AT_HP_opt_level";
        break;
      case 0x2015u:
        result = "DW_AT_HP_prof_version_id";
        break;
      case 0x2016u:
        result = "DW_AT_HP_opt_flags";
        break;
      case 0x2017u:
        result = "DW_AT_HP_cold_region_low_pc";
        break;
      case 0x2018u:
        result = "DW_AT_HP_cold_region_high_pc";
        break;
      case 0x2019u:
        result = "DW_AT_HP_all_variables_modifiable";
        break;
      case 0x201Au:
        result = "DW_AT_HP_linkage_name";
        break;
      case 0x201Bu:
        result = "DW_AT_HP_prof_flags";
        break;
      case 0x201Fu:
        result = "DW_AT_HP_unit_name";
        break;
      case 0x2020u:
        result = "DW_AT_HP_unit_size";
        break;
      case 0x2021u:
        result = "DW_AT_HP_widened_byte_size";
        break;
      case 0x2022u:
        result = "DW_AT_HP_definition_points";
        break;
      case 0x2023u:
        result = "DW_AT_HP_default_location";
        break;
      case 0x2026u:
        result = "DW_AT_DW_AT_INTEL_other_endian";
        break;
      case 0x2029u:
        result = "DW_AT_HP_is_result_param";
        break;
      case 0x2083u:
        result = "DW_AT_GHS_rsm";
        break;
      case 0x2085u:
        result = "DW_AT_GHS_frsm";
        break;
      case 0x2086u:
        result = "DW_AT_GHS_frames";
        break;
      case 0x2087u:
        result = "DW_AT_GHS_rso";
        break;
      case 0x2092u:
        result = "DW_AT_GHS_subcpu";
        break;
      case 0x2093u:
        result = "DW_AT_GHS_lbrace_line";
        break;
      case 0x2101u:
        result = "DW_AT_sf_names";
        break;
      case 0x2102u:
        result = "DW_AT_src_info";
        break;
      case 0x2103u:
        result = "DW_AT_mac_info";
        break;
      case 0x2104u:
        result = "DW_AT_src_coords";
        break;
      case 0x2105u:
        result = "DW_AT_body_begin";
        break;
      case 0x2106u:
        result = "DW_AT_body_end";
        break;
      case 0x2107u:
        result = "DW_AT_GNU_vector";
        break;
      case 0x210Fu:
        result = "DW_AT_GNU_odr_signature";
        break;
      case 0x2110u:
        result = "DW_AT_GNU_template_name";
        break;
      case 0x2111u:
        result = "DW_AT_GNU_call_site_value";
        break;
      case 0x2112u:
        result = "DW_AT_GNU_call_site_data_value";
        break;
      case 0x2113u:
        result = "DW_AT_GNU_call_site_target";
        break;
      case 0x2114u:
        result = "DW_AT_GNU_call_site_target_clobbered";
        break;
      case 0x2115u:
        result = "DW_AT_GNU_tail_call";
        break;
      case 0x2116u:
        result = "DW_AT_GNU_all_tail_call_sites";
        break;
      case 0x2117u:
        result = "DW_AT_GNU_all_call_sites";
        break;
      case 0x2118u:
        result = "DW_AT_GNU_all_source_call_sites";
        break;
      case 0x2119u:
        result = "DW_AT_GNU_macros";
        break;
      case 0x211Au:
        result = "DW_AT_GNU_deleted";
        break;
      case 0x2130u:
        result = "DW_AT_GNU_dwo_name";
        break;
      case 0x2131u:
        result = "DW_AT_GNU_dwo_id";
        break;
      case 0x2132u:
        result = "DW_AT_GNU_ranges_base";
        break;
      case 0x2133u:
        result = "DW_AT_GNU_addr_base";
        break;
      case 0x2134u:
        result = "DW_AT_GNU_pubnames";
        break;
      case 0x2135u:
        result = "DW_AT_GNU_pubtypes";
        break;
      case 0x2136u:
        result = "DW_AT_GNU_discriminator";
        break;
      case 0x2137u:
        result = "DW_AT_GNU_locviews";
        break;
      case 0x2138u:
        result = "DW_AT_GNU_entry_view";
        break;
      case 0x2201u:
        result = "DW_AT_SUN_template";
        break;
      case 0x2202u:
        result = "DW_AT_SUN_alignment";
        break;
      case 0x2203u:
        result = "DW_AT_SUN_vtable";
        break;
      case 0x2204u:
        result = "DW_AT_SUN_count_guarantee";
        break;
      case 0x2205u:
        result = "DW_AT_SUN_command_line";
        break;
      case 0x2206u:
        result = "DW_AT_SUN_vbase";
        break;
      case 0x2207u:
        result = "DW_AT_SUN_compile_options";
        break;
      case 0x2208u:
        result = "DW_AT_SUN_language";
        break;
      case 0x2209u:
        result = "DW_AT_SUN_browser_file";
        break;
      case 0x2210u:
        result = "DW_AT_SUN_vtable_abi";
        break;
      case 0x2211u:
        result = "DW_AT_SUN_func_offsets";
        break;
      case 0x2212u:
        result = "DW_AT_SUN_cf_kind";
        break;
      case 0x2213u:
        result = "DW_AT_SUN_vtable_index";
        break;
      case 0x2214u:
        result = "DW_AT_SUN_omp_tpriv_addr";
        break;
      case 0x2215u:
        result = "DW_AT_SUN_omp_child_func";
        break;
      case 0x2216u:
        result = "DW_AT_SUN_func_offset";
        break;
      case 0x2217u:
        result = "DW_AT_SUN_memop_type_ref";
        break;
      case 0x2218u:
        result = "DW_AT_SUN_profile_id";
        break;
      case 0x2219u:
        result = "DW_AT_SUN_memop_signature";
        break;
      case 0x2220u:
        result = "DW_AT_SUN_obj_dir";
        break;
      case 0x2221u:
        result = "DW_AT_SUN_obj_file";
        break;
      case 0x2222u:
        result = "DW_AT_SUN_original_name";
        break;
      case 0x2223u:
        result = "DW_AT_SUN_hwcprof_signature";
        break;
      case 0x2224u:
        result = "DW_AT_SUN_amd64_parmdump";
        break;
      case 0x2225u:
        result = "DW_AT_SUN_part_link_name";
        break;
      case 0x2226u:
        result = "DW_AT_SUN_link_name";
        break;
      case 0x2227u:
        result = "DW_AT_SUN_pass_with_const";
        break;
      case 0x2228u:
        result = "DW_AT_SUN_return_with_const";
        break;
      case 0x2229u:
        result = "DW_AT_SUN_import_by_name";
        break;
      case 0x222Au:
        result = "DW_AT_SUN_90_pointer";
        break;
      case 0x222Bu:
        result = "DW_AT_SUN_pass_by_ref";
        break;
      case 0x222Cu:
        result = "DW_AT_SUN_f90_allocatable";
        break;
      case 0x222Du:
        result = "DW_AT_SUN_f90_assumed_shape_array";
        break;
      case 0x222Eu:
        result = "DW_AT_SUN_c_vla";
        break;
      case 0x2230u:
        result = "DW_AT_SUN_return_value_ptr";
        break;
      case 0x2231u:
        result = "DW_AT_SUN_dtor_start";
        break;
      case 0x2232u:
        result = "DW_AT_SUN_dtor_length";
        break;
      case 0x2233u:
        result = "DW_AT_SUN_dtor_state_initial";
        break;
      case 0x2234u:
        result = "DW_AT_SUN_dtor_state_final";
        break;
      case 0x2235u:
        result = "DW_AT_SUN_dtor_state_deltas";
        break;
      case 0x2236u:
        result = "DW_AT_SUN_import_by_lname";
        break;
      case 0x2237u:
        result = "DW_AT_SUN_f90_use_only";
        break;
      case 0x2238u:
        result = "DW_AT_SUN_namelist_spec";
        break;
      case 0x2239u:
        result = "DW_AT_SUN_is_omp_child_func";
        break;
      case 0x223Au:
        result = "DW_AT_SUN_fortran_main_alias";
        break;
      case 0x223Bu:
        result = "DW_AT_SUN_fortran_based";
        break;
      case 0x2300u:
        result = "DW_AT_ALTIUM_loclist";
        break;
      case 0x2301u:
        result = "DW_AT_use_GNAT_descriptive_type";
        break;
      case 0x2302u:
        result = "DW_AT_GNAT_descriptive_type";
        break;
      case 0x2303u:
        result = "DW_AT_GNU_numerator";
        break;
      case 0x2304u:
        result = "DW_AT_GNU_denominator";
        break;
      case 0x2305u:
        result = "DW_AT_GNU_bias";
        break;
      default:
        return 0;
    }
  }
  return result;
}
