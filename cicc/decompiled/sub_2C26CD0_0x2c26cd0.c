// Function: sub_2C26CD0
// Address: 0x2c26cd0
//
__int64 __fastcall sub_2C26CD0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // kr00_8

  v3 = v1;
  result = *(unsigned __int8 *)(a1 + 152);
  switch ( *(_BYTE *)(a1 + 152) )
  {
    case 1:
      *(_BYTE *)(a1 + 156) &= 0xFCu;
      break;
    case 2:
    case 3:
    case 6:
      *(_BYTE *)(a1 + 156) &= ~1u;
      break;
    case 4:
      *(_DWORD *)(a1 + 156) = 0;
      break;
    case 5:
      *(_BYTE *)(a1 + 156) &= 0xF9u;
      break;
    default:
      result = v3;
      break;
  }
  return result;
}
