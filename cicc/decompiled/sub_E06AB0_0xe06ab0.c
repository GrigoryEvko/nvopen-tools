// Function: sub_E06AB0
// Address: 0xe06ab0
//
const char *__fastcall sub_E06AB0(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 0x2C )
  {
    if ( a1 == 7968 )
      return "DW_FORM_GNU_ref_alt";
    if ( a1 <= 0x1F20 )
    {
      result = "DW_FORM_GNU_addr_index";
      if ( a1 == 7937 )
        return result;
      if ( a1 == 7938 )
        return "DW_FORM_GNU_str_index";
    }
    else
    {
      result = "DW_FORM_GNU_strp_alt";
      if ( a1 == 7969 )
        return result;
      if ( a1 == 8193 )
        return "DW_FORM_LLVM_addrx_offset";
    }
    return 0;
  }
  if ( !a1 )
    return 0;
  switch ( a1 )
  {
    case 1u:
      result = "DW_FORM_addr";
      break;
    case 3u:
      result = "DW_FORM_block2";
      break;
    case 4u:
      result = "DW_FORM_block4";
      break;
    case 5u:
      result = "DW_FORM_data2";
      break;
    case 6u:
      result = "DW_FORM_data4";
      break;
    case 7u:
      result = "DW_FORM_data8";
      break;
    case 8u:
      result = "DW_FORM_string";
      break;
    case 9u:
      result = "DW_FORM_block";
      break;
    case 0xAu:
      result = "DW_FORM_block1";
      break;
    case 0xBu:
      result = "DW_FORM_data1";
      break;
    case 0xCu:
      result = "DW_FORM_flag";
      break;
    case 0xDu:
      result = "DW_FORM_sdata";
      break;
    case 0xEu:
      result = "DW_FORM_strp";
      break;
    case 0xFu:
      result = "DW_FORM_udata";
      break;
    case 0x10u:
      result = "DW_FORM_ref_addr";
      break;
    case 0x11u:
      result = "DW_FORM_ref1";
      break;
    case 0x12u:
      result = "DW_FORM_ref2";
      break;
    case 0x13u:
      result = "DW_FORM_ref4";
      break;
    case 0x14u:
      result = "DW_FORM_ref8";
      break;
    case 0x15u:
      result = "DW_FORM_ref_udata";
      break;
    case 0x16u:
      result = "DW_FORM_indirect";
      break;
    case 0x17u:
      result = "DW_FORM_sec_offset";
      break;
    case 0x18u:
      result = "DW_FORM_exprloc";
      break;
    case 0x19u:
      result = "DW_FORM_flag_present";
      break;
    case 0x1Au:
      result = "DW_FORM_strx";
      break;
    case 0x1Bu:
      result = "DW_FORM_addrx";
      break;
    case 0x1Cu:
      result = "DW_FORM_ref_sup4";
      break;
    case 0x1Du:
      result = "DW_FORM_strp_sup";
      break;
    case 0x1Eu:
      result = "DW_FORM_data16";
      break;
    case 0x1Fu:
      result = "DW_FORM_line_strp";
      break;
    case 0x20u:
      result = "DW_FORM_ref_sig8";
      break;
    case 0x21u:
      result = "DW_FORM_implicit_const";
      break;
    case 0x22u:
      result = "DW_FORM_loclistx";
      break;
    case 0x23u:
      result = "DW_FORM_rnglistx";
      break;
    case 0x24u:
      result = "DW_FORM_ref_sup8";
      break;
    case 0x25u:
      result = "DW_FORM_strx1";
      break;
    case 0x26u:
      result = "DW_FORM_strx2";
      break;
    case 0x27u:
      result = "DW_FORM_strx3";
      break;
    case 0x28u:
      result = "DW_FORM_strx4";
      break;
    case 0x29u:
      result = "DW_FORM_addrx1";
      break;
    case 0x2Au:
      result = "DW_FORM_addrx2";
      break;
    case 0x2Bu:
      result = "DW_FORM_addrx3";
      break;
    case 0x2Cu:
      result = "DW_FORM_addrx4";
      break;
    default:
      return 0;
  }
  return result;
}
