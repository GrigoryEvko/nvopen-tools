// Function: sub_5E6440
// Address: 0x5e6440
//
__int64 *__fastcall sub_5E6440(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 i; // rbx
  __int64 v7; // rsi
  __int64 v9; // rax

  v3 = a1;
  if ( a2 )
    v3 = *(_QWORD *)(a2 + 40);
  for ( i = *(_QWORD *)(*(_QWORD *)(v3 + 168) + 8LL); i; i = *(_QWORD *)(i + 8) )
  {
    v7 = i;
    if ( a2 )
      v7 = sub_8E5650(i);
    if ( (*(_BYTE *)(v7 + 96) & 2) != 0 )
    {
      v9 = *(_QWORD *)(a1 + 168);
      while ( 1 )
      {
        v9 = *(_QWORD *)(v9 + 16);
        if ( !v9 )
          break;
        if ( v7 == v9 )
          goto LABEL_8;
      }
    }
    *a3 = v7;
    a3 = (__int64 *)sub_5E6440(a1, v7, v7 + 16);
LABEL_8:
    ;
  }
  return a3;
}
