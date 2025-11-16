// Function: sub_E06E20
// Address: 0xe06e20
//
const char *__fastcall sub_E06E20(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 0xFC )
  {
    switch ( a1 )
    {
      case 0x1000u:
        result = "DW_OP_LLVM_fragment";
        break;
      case 0x1001u:
        result = "DW_OP_LLVM_convert";
        break;
      case 0x1002u:
        result = "DW_OP_LLVM_tag_offset";
        break;
      case 0x1003u:
        result = "DW_OP_LLVM_entry_value";
        break;
      case 0x1004u:
        result = "DW_OP_LLVM_implicit_pointer";
        break;
      case 0x1005u:
        result = "DW_OP_LLVM_arg";
        break;
      case 0x1006u:
        result = "DW_OP_LLVM_extract_bits_sext";
        break;
      case 0x1007u:
        result = "DW_OP_LLVM_extract_bits_zext";
        break;
      default:
        return 0;
    }
  }
  else if ( a1 <= 2 )
  {
    return 0;
  }
  else
  {
    switch ( a1 )
    {
      case 3u:
        result = "DW_OP_addr";
        break;
      case 6u:
        result = "DW_OP_deref";
        break;
      case 8u:
        result = "DW_OP_const1u";
        break;
      case 9u:
        result = "DW_OP_const1s";
        break;
      case 0xAu:
        result = "DW_OP_const2u";
        break;
      case 0xBu:
        result = "DW_OP_const2s";
        break;
      case 0xCu:
        result = "DW_OP_const4u";
        break;
      case 0xDu:
        result = "DW_OP_const4s";
        break;
      case 0xEu:
        result = "DW_OP_const8u";
        break;
      case 0xFu:
        result = "DW_OP_const8s";
        break;
      case 0x10u:
        result = "DW_OP_constu";
        break;
      case 0x11u:
        result = "DW_OP_consts";
        break;
      case 0x12u:
        result = "DW_OP_dup";
        break;
      case 0x13u:
        result = "DW_OP_drop";
        break;
      case 0x14u:
        result = "DW_OP_over";
        break;
      case 0x15u:
        result = "DW_OP_pick";
        break;
      case 0x16u:
        result = "DW_OP_swap";
        break;
      case 0x17u:
        result = "DW_OP_rot";
        break;
      case 0x18u:
        result = "DW_OP_xderef";
        break;
      case 0x19u:
        result = "DW_OP_abs";
        break;
      case 0x1Au:
        result = "DW_OP_and";
        break;
      case 0x1Bu:
        result = "DW_OP_div";
        break;
      case 0x1Cu:
        result = "DW_OP_minus";
        break;
      case 0x1Du:
        result = "DW_OP_mod";
        break;
      case 0x1Eu:
        result = "DW_OP_mul";
        break;
      case 0x1Fu:
        result = "DW_OP_neg";
        break;
      case 0x20u:
        result = "DW_OP_not";
        break;
      case 0x21u:
        result = "DW_OP_or";
        break;
      case 0x22u:
        result = "DW_OP_plus";
        break;
      case 0x23u:
        result = "DW_OP_plus_uconst";
        break;
      case 0x24u:
        result = "DW_OP_shl";
        break;
      case 0x25u:
        result = "DW_OP_shr";
        break;
      case 0x26u:
        result = "DW_OP_shra";
        break;
      case 0x27u:
        result = "DW_OP_xor";
        break;
      case 0x28u:
        result = "DW_OP_bra";
        break;
      case 0x29u:
        result = "DW_OP_eq";
        break;
      case 0x2Au:
        result = "DW_OP_ge";
        break;
      case 0x2Bu:
        result = "DW_OP_gt";
        break;
      case 0x2Cu:
        result = "DW_OP_le";
        break;
      case 0x2Du:
        result = "DW_OP_lt";
        break;
      case 0x2Eu:
        result = "DW_OP_ne";
        break;
      case 0x2Fu:
        result = "DW_OP_skip";
        break;
      case 0x30u:
        result = "DW_OP_lit0";
        break;
      case 0x31u:
        result = "DW_OP_lit1";
        break;
      case 0x32u:
        result = "DW_OP_lit2";
        break;
      case 0x33u:
        result = "DW_OP_lit3";
        break;
      case 0x34u:
        result = "DW_OP_lit4";
        break;
      case 0x35u:
        result = "DW_OP_lit5";
        break;
      case 0x36u:
        result = "DW_OP_lit6";
        break;
      case 0x37u:
        result = "DW_OP_lit7";
        break;
      case 0x38u:
        result = "DW_OP_lit8";
        break;
      case 0x39u:
        result = "DW_OP_lit9";
        break;
      case 0x3Au:
        result = "DW_OP_lit10";
        break;
      case 0x3Bu:
        result = "DW_OP_lit11";
        break;
      case 0x3Cu:
        result = "DW_OP_lit12";
        break;
      case 0x3Du:
        result = "DW_OP_lit13";
        break;
      case 0x3Eu:
        result = "DW_OP_lit14";
        break;
      case 0x3Fu:
        result = "DW_OP_lit15";
        break;
      case 0x40u:
        result = "DW_OP_lit16";
        break;
      case 0x41u:
        result = "DW_OP_lit17";
        break;
      case 0x42u:
        result = "DW_OP_lit18";
        break;
      case 0x43u:
        result = "DW_OP_lit19";
        break;
      case 0x44u:
        result = "DW_OP_lit20";
        break;
      case 0x45u:
        result = "DW_OP_lit21";
        break;
      case 0x46u:
        result = "DW_OP_lit22";
        break;
      case 0x47u:
        result = "DW_OP_lit23";
        break;
      case 0x48u:
        result = "DW_OP_lit24";
        break;
      case 0x49u:
        result = "DW_OP_lit25";
        break;
      case 0x4Au:
        result = "DW_OP_lit26";
        break;
      case 0x4Bu:
        result = "DW_OP_lit27";
        break;
      case 0x4Cu:
        result = "DW_OP_lit28";
        break;
      case 0x4Du:
        result = "DW_OP_lit29";
        break;
      case 0x4Eu:
        result = "DW_OP_lit30";
        break;
      case 0x4Fu:
        result = "DW_OP_lit31";
        break;
      case 0x50u:
        result = "DW_OP_reg0";
        break;
      case 0x51u:
        result = "DW_OP_reg1";
        break;
      case 0x52u:
        result = "DW_OP_reg2";
        break;
      case 0x53u:
        result = "DW_OP_reg3";
        break;
      case 0x54u:
        result = "DW_OP_reg4";
        break;
      case 0x55u:
        result = "DW_OP_reg5";
        break;
      case 0x56u:
        result = "DW_OP_reg6";
        break;
      case 0x57u:
        result = "DW_OP_reg7";
        break;
      case 0x58u:
        result = "DW_OP_reg8";
        break;
      case 0x59u:
        result = "DW_OP_reg9";
        break;
      case 0x5Au:
        result = "DW_OP_reg10";
        break;
      case 0x5Bu:
        result = "DW_OP_reg11";
        break;
      case 0x5Cu:
        result = "DW_OP_reg12";
        break;
      case 0x5Du:
        result = "DW_OP_reg13";
        break;
      case 0x5Eu:
        result = "DW_OP_reg14";
        break;
      case 0x5Fu:
        result = "DW_OP_reg15";
        break;
      case 0x60u:
        result = "DW_OP_reg16";
        break;
      case 0x61u:
        result = "DW_OP_reg17";
        break;
      case 0x62u:
        result = "DW_OP_reg18";
        break;
      case 0x63u:
        result = "DW_OP_reg19";
        break;
      case 0x64u:
        result = "DW_OP_reg20";
        break;
      case 0x65u:
        result = "DW_OP_reg21";
        break;
      case 0x66u:
        result = "DW_OP_reg22";
        break;
      case 0x67u:
        result = "DW_OP_reg23";
        break;
      case 0x68u:
        result = "DW_OP_reg24";
        break;
      case 0x69u:
        result = "DW_OP_reg25";
        break;
      case 0x6Au:
        result = "DW_OP_reg26";
        break;
      case 0x6Bu:
        result = "DW_OP_reg27";
        break;
      case 0x6Cu:
        result = "DW_OP_reg28";
        break;
      case 0x6Du:
        result = "DW_OP_reg29";
        break;
      case 0x6Eu:
        result = "DW_OP_reg30";
        break;
      case 0x6Fu:
        result = "DW_OP_reg31";
        break;
      case 0x70u:
        result = "DW_OP_breg0";
        break;
      case 0x71u:
        result = "DW_OP_breg1";
        break;
      case 0x72u:
        result = "DW_OP_breg2";
        break;
      case 0x73u:
        result = "DW_OP_breg3";
        break;
      case 0x74u:
        result = "DW_OP_breg4";
        break;
      case 0x75u:
        result = "DW_OP_breg5";
        break;
      case 0x76u:
        result = "DW_OP_breg6";
        break;
      case 0x77u:
        result = "DW_OP_breg7";
        break;
      case 0x78u:
        result = "DW_OP_breg8";
        break;
      case 0x79u:
        result = "DW_OP_breg9";
        break;
      case 0x7Au:
        result = "DW_OP_breg10";
        break;
      case 0x7Bu:
        result = "DW_OP_breg11";
        break;
      case 0x7Cu:
        result = "DW_OP_breg12";
        break;
      case 0x7Du:
        result = "DW_OP_breg13";
        break;
      case 0x7Eu:
        result = "DW_OP_breg14";
        break;
      case 0x7Fu:
        result = "DW_OP_breg15";
        break;
      case 0x80u:
        result = "DW_OP_breg16";
        break;
      case 0x81u:
        result = "DW_OP_breg17";
        break;
      case 0x82u:
        result = "DW_OP_breg18";
        break;
      case 0x83u:
        result = "DW_OP_breg19";
        break;
      case 0x84u:
        result = "DW_OP_breg20";
        break;
      case 0x85u:
        result = "DW_OP_breg21";
        break;
      case 0x86u:
        result = "DW_OP_breg22";
        break;
      case 0x87u:
        result = "DW_OP_breg23";
        break;
      case 0x88u:
        result = "DW_OP_breg24";
        break;
      case 0x89u:
        result = "DW_OP_breg25";
        break;
      case 0x8Au:
        result = "DW_OP_breg26";
        break;
      case 0x8Bu:
        result = "DW_OP_breg27";
        break;
      case 0x8Cu:
        result = "DW_OP_breg28";
        break;
      case 0x8Du:
        result = "DW_OP_breg29";
        break;
      case 0x8Eu:
        result = "DW_OP_breg30";
        break;
      case 0x8Fu:
        result = "DW_OP_breg31";
        break;
      case 0x90u:
        result = "DW_OP_regx";
        break;
      case 0x91u:
        result = "DW_OP_fbreg";
        break;
      case 0x92u:
        result = "DW_OP_bregx";
        break;
      case 0x93u:
        result = "DW_OP_piece";
        break;
      case 0x94u:
        result = "DW_OP_deref_size";
        break;
      case 0x95u:
        result = "DW_OP_xderef_size";
        break;
      case 0x96u:
        result = "DW_OP_nop";
        break;
      case 0x97u:
        result = "DW_OP_push_object_address";
        break;
      case 0x98u:
        result = "DW_OP_call2";
        break;
      case 0x99u:
        result = "DW_OP_call4";
        break;
      case 0x9Au:
        result = "DW_OP_call_ref";
        break;
      case 0x9Bu:
        result = "DW_OP_form_tls_address";
        break;
      case 0x9Cu:
        result = "DW_OP_call_frame_cfa";
        break;
      case 0x9Du:
        result = "DW_OP_bit_piece";
        break;
      case 0x9Eu:
        result = "DW_OP_implicit_value";
        break;
      case 0x9Fu:
        result = "DW_OP_stack_value";
        break;
      case 0xA0u:
        result = "DW_OP_implicit_pointer";
        break;
      case 0xA1u:
        result = "DW_OP_addrx";
        break;
      case 0xA2u:
        result = "DW_OP_constx";
        break;
      case 0xA3u:
        result = "DW_OP_entry_value";
        break;
      case 0xA4u:
        result = "DW_OP_const_type";
        break;
      case 0xA5u:
        result = "DW_OP_regval_type";
        break;
      case 0xA6u:
        result = "DW_OP_deref_type";
        break;
      case 0xA7u:
        result = "DW_OP_xderef_type";
        break;
      case 0xA8u:
        result = "DW_OP_convert";
        break;
      case 0xA9u:
        result = "DW_OP_reinterpret";
        break;
      case 0xE0u:
        result = "DW_OP_GNU_push_tls_address";
        break;
      case 0xE1u:
        result = "DW_OP_HP_is_value";
        break;
      case 0xE2u:
        result = "DW_OP_HP_fltconst4";
        break;
      case 0xE3u:
        result = "DW_OP_HP_fltconst8";
        break;
      case 0xE4u:
        result = "DW_OP_HP_mod_range";
        break;
      case 0xE5u:
        result = "DW_OP_HP_unmod_range";
        break;
      case 0xE6u:
        result = "DW_OP_HP_tls";
        break;
      case 0xE8u:
        result = "DW_OP_INTEL_bit_piece";
        break;
      case 0xE9u:
        result = "DW_OP_LLVM_user";
        break;
      case 0xEDu:
        result = "DW_OP_WASM_location";
        break;
      case 0xEEu:
        result = "DW_OP_WASM_location_int";
        break;
      case 0xF0u:
        result = "DW_OP_APPLE_uninit";
        break;
      case 0xF3u:
        result = "DW_OP_GNU_entry_value";
        break;
      case 0xF8u:
        result = "DW_OP_PGI_omp_thread_num";
        break;
      case 0xFBu:
        result = "DW_OP_GNU_addr_index";
        break;
      case 0xFCu:
        result = "DW_OP_GNU_const_index";
        break;
      default:
        return 0;
    }
  }
  return result;
}
