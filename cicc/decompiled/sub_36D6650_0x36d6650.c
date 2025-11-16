// Function: sub_36D6650
// Address: 0x36d6650
//
__int64 __fastcall sub_36D6650(unsigned __int16 a1, int a2, int a3, int a4, __int64 a5, int a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // [rsp+10h] [rbp-8h]

  if ( a1 > 0x2Fu )
  {
    if ( a1 == 127 || a1 == 138 )
    {
LABEL_6:
      LODWORD(v8) = a4;
      BYTE4(v8) = 1;
      return v8;
    }
    goto LABEL_8;
  }
  if ( a1 <= 1u )
  {
LABEL_8:
    BYTE4(v8) = 0;
    return v8;
  }
  switch ( a1 )
  {
    case 2u:
    case 5u:
      LODWORD(v8) = a2;
      BYTE4(v8) = 1;
      result = v8;
      break;
    case 6u:
    case 0xAu:
    case 0xBu:
      LODWORD(v8) = a3;
      BYTE4(v8) = 1;
      result = v8;
      break;
    case 7u:
    case 0x25u:
    case 0x2Fu:
      goto LABEL_6;
    case 8u:
      result = a5;
      break;
    case 0xCu:
      LODWORD(v8) = a6;
      BYTE4(v8) = 1;
      result = v8;
      break;
    case 0xDu:
      result = a7;
      break;
    default:
      goto LABEL_8;
  }
  return result;
}
