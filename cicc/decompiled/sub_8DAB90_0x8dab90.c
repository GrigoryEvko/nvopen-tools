// Function: sub_8DAB90
// Address: 0x8dab90
//
__int64 __fastcall sub_8DAB90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 i; // rdi

  v5 = a2;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    a1 = *(_QWORD *)(a1 + 160);
  while ( *(_BYTE *)(a1 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
  v6 = *(_QWORD *)(a1 + 160);
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    for ( i = *(_QWORD *)(v6 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( v5 == i || (unsigned int)sub_8D97D0(i, v5, 0, a4, a5) )
      break;
    v6 = *(_QWORD *)(v6 + 112);
    if ( !v6 )
      return 0;
  }
  return 1;
}
