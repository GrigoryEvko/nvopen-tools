// Function: sub_808FB0
// Address: 0x808fb0
//
__int64 __fastcall sub_808FB0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 result; // rax

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v2 = *(_QWORD *)(a1 + 160);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    result = *(_QWORD *)(v2 + 8);
    if ( result )
      break;
    if ( (*(_BYTE *)(v2 + 144) & 0x10) != 0 )
    {
      result = sub_808FB0(*(_QWORD *)(v2 + 120), a2);
      if ( result )
        return result;
    }
    v2 = *(_QWORD *)(v2 + 112);
    if ( !v2 )
      return 0;
  }
  *a2 = v2;
  return result;
}
