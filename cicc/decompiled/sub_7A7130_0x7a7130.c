// Function: sub_7A7130
// Address: 0x7a7130
//
__int64 __fastcall sub_7A7130(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8

  for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 144) & 4) == 0 )
      break;
    if ( *(_BYTE *)(i + 137) )
      break;
  }
  if ( a2 != i )
    return 0;
  v6 = **(_QWORD ***)(a1 + 168);
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    if ( a3 == v6[13] )
    {
      v7 = v6[5];
      if ( (*(_BYTE *)(v7 + 179) & 1) != 0 && !**(_QWORD **)(v7 + 168) )
      {
        v8 = sub_7A6790(*(_QWORD *)(a2 + 120));
        if ( (unsigned int)sub_7A6F40(v8, (__int64)v6, 0, 0, v9) )
          break;
      }
    }
    v6 = (_QWORD *)*v6;
    if ( !v6 )
      return 0;
  }
  return 1;
}
