// Function: sub_31F08F0
// Address: 0x31f08f0
//
char *__fastcall sub_31F08F0(unsigned int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0u:
      result = "absptr";
      break;
    case 1u:
      result = "uleb128";
      break;
    case 2u:
    case 5u:
    case 6u:
    case 7u:
    case 8u:
    case 0xAu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x11u:
    case 0x12u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Au:
      goto LABEL_2;
    case 3u:
      result = "udata4";
      break;
    case 4u:
      result = "udata8";
      break;
    case 9u:
      result = "sleb128";
      break;
    case 0xBu:
      result = "sdata4";
      break;
    case 0xCu:
      result = "sdata8";
      break;
    case 0x10u:
      result = "pcrel";
      break;
    case 0x13u:
      result = "pcrel udata4";
      break;
    case 0x14u:
      result = "pcrel udata8";
      break;
    case 0x1Bu:
      result = "pcrel sdata4";
      break;
    case 0x1Cu:
      result = "pcrel sdata8";
      break;
    default:
      if ( a1 > 0xBC )
      {
        result = "<unknown encoding>";
        if ( a1 == 255 )
          result = "omit";
      }
      else if ( a1 <= 0x92 )
      {
LABEL_2:
        result = "<unknown encoding>";
      }
      else
      {
        switch ( a1 )
        {
          case 0x93u:
            result = "indirect pcrel udata4";
            break;
          case 0x94u:
            result = "indirect pcrel udata8";
            break;
          case 0x9Bu:
            result = "indirect pcrel sdata4";
            break;
          case 0x9Cu:
            result = "indirect pcrel sdata8";
            break;
          case 0xBBu:
            result = "indirect datarel sdata4";
            break;
          case 0xBCu:
            result = "indirect datarel sdata8";
            break;
          default:
            goto LABEL_2;
        }
      }
      break;
  }
  return result;
}
