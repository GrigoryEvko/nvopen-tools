// Function: sub_8C3020
// Address: 0x8c3020
//
void __fastcall sub_8C3020(__int64 a1)
{
  __int64 i; // r12
  __int64 v2; // rbx
  __int64 j; // rbx
  _QWORD *k; // rbx
  _QWORD *v5; // rbx

  for ( i = *(_QWORD *)(a1 + 144); i; i = *(_QWORD *)(i + 112) )
  {
    while ( !*(_DWORD *)(i + 160) )
    {
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_8;
    }
    v2 = sub_72B840(i);
    sub_75AFC0(
      *(_DWORD *)(i + 164),
      (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C2C50,
      (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))sub_8C3290,
      0,
      0,
      (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C37B0,
      0);
    do
    {
      *(_BYTE *)(v2 + 29) &= ~1u;
      v2 = *(_QWORD *)v2;
    }
    while ( v2 );
  }
LABEL_8:
  if ( dword_4F077C4 == 2 )
    sub_8C3170(*(_QWORD *)(a1 + 104));
  for ( j = *(_QWORD *)(a1 + 168); j; j = *(_QWORD *)(j + 112) )
  {
    if ( (*(_BYTE *)(j + 124) & 1) == 0 )
      sub_8C3020(*(_QWORD *)(j + 128));
  }
  for ( k = *(_QWORD **)(a1 + 160); k; k = (_QWORD *)*k )
    sub_8C3020(k);
  if ( dword_4F077C4 == 2 && !*(_BYTE *)(a1 + 28) )
  {
    v5 = (_QWORD *)qword_4F072C0;
    if ( qword_4F072C0 )
    {
      do
      {
        sub_8C3170(v5[3]);
        v5 = (_QWORD *)*v5;
      }
      while ( v5 );
    }
  }
}
