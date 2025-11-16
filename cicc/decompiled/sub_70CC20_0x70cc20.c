// Function: sub_70CC20
// Address: 0x70cc20
//
__int64 __fastcall sub_70CC20(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 173) == 1 )
    return 0;
  switch ( *(_BYTE *)(a1 + 176) )
  {
    case 0:
      result = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(result + 200) & 0x20) == 0 )
        return result;
      goto LABEL_5;
    case 1:
      result = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(result + 168) & 8) != 0 )
LABEL_5:
        *a2 = 1;
      break;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      result = *(_QWORD *)(a1 + 184);
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
