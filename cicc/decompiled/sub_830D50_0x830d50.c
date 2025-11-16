// Function: sub_830D50
// Address: 0x830d50
//
__int64 __fastcall sub_830D50(__int64 a1, __int64 a2, __int64 *a3, int a4, __int64 a5)
{
  __int64 v7; // r14
  unsigned int v9; // r15d
  __int64 i; // rdi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  char k; // dl
  int v17; // eax
  __int64 *v18; // rax
  char v19; // al
  int v20; // esi
  __int64 v21; // rdi
  const __m128i *j; // rdi
  __m128i *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v27; // [rsp+10h] [rbp-40h] BYREF
  __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
  {
    v7 = a1;
    if ( *(_BYTE *)(a1 + 80) == 8 && *(char *)(*(_QWORD *)(a1 + 88) + 144LL) < 0 )
    {
      v17 = sub_85E8D0();
      v9 = 1;
      v27 = (__int64 *)sub_8309B0(*(_QWORD **)(qword_4F04C68[0] + 776LL * v17 + 184));
      v18 = sub_73E830((__int64)v27);
      sub_6E70E0(v18, a5);
      goto LABEL_8;
    }
    v9 = sub_830940(&v27, v28);
    if ( !v9 )
      goto LABEL_5;
    for ( i = sub_8D46C0(v28[0]); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v12 = *(_QWORD *)(a2 + 64);
    if ( v12 == i || v12 && dword_4F07588 && (v13 = *(_QWORD *)(i + 32), *(_QWORD *)(v12 + 32) == v13) && v13 )
    {
      v9 = 1;
    }
    else
    {
      v14 = sub_8D5CE0(i, v12);
      v9 = v14 != 0;
      if ( !v14 )
      {
        if ( dword_4F04C44 == -1 )
        {
          v25 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( (*(_BYTE *)(v25 + 6) & 6) == 0 && *(_BYTE *)(v25 + 4) != 12 )
          {
LABEL_5:
            if ( (unsigned int)sub_6E5430() )
              sub_6851C0(0xF5u, a3);
            sub_6E6260((_QWORD *)a5);
            goto LABEL_8;
          }
        }
        sub_830B80(v27, v28[0], 1, a3, a3, (_QWORD *)a5);
        v19 = *(_BYTE *)(v7 + 80);
        if ( v19 == 16 )
        {
          v7 = **(_QWORD **)(v7 + 88);
          v19 = *(_BYTE *)(v7 + 80);
        }
        if ( v19 == 24 )
          v7 = *(_QWORD *)(v7 + 88);
        v20 = 0;
        v21 = sub_8D46C0(v28[0]);
        if ( (*(_BYTE *)(v21 + 140) & 0xFB) == 8 )
          v20 = sub_8D4C10(v21, dword_4F077C4 != 2);
        for ( j = *(const __m128i **)(v7 + 64); j[8].m128i_i8[12] == 12; j = (const __m128i *)j[10].m128i_i64[0] )
          ;
        v9 = 1;
        v23 = sub_73C570(j, v20);
        v24 = sub_72D2E0(v23);
        sub_6FB850(v24, (__m128i *)a5, 0, 0, 0, 1, 0, 0);
        goto LABEL_18;
      }
    }
    sub_830B80(v27, v28[0], 1, a3, a3, (_QWORD *)a5);
    sub_82FD20((const __m128i *)a5, 1, v7, a2, a4, 0, (__int64)a3);
LABEL_18:
    if ( !*(_BYTE *)(a5 + 16) )
      goto LABEL_22;
    v15 = *(_QWORD *)a5;
    for ( k = *(_BYTE *)(*(_QWORD *)a5 + 140LL); k == 12; k = *(_BYTE *)(v15 + 140) )
      v15 = *(_QWORD *)(v15 + 160);
    if ( k )
    {
      if ( !(dword_4F077BC | (unsigned int)sub_6E4B50()) && (unsigned int)sub_6E91E0(0x983u, a3) )
      {
        v9 = 0;
        sub_6E6840(a5);
      }
    }
    else
    {
LABEL_22:
      v9 = 0;
    }
    goto LABEL_8;
  }
  if ( (unsigned int)sub_6E5430() )
    sub_6851C0(0x1Cu, a3);
  v9 = 0;
  sub_6E6260((_QWORD *)a5);
LABEL_8:
  *(_QWORD *)(a5 + 68) = *a3;
  sub_6E26D0(2, a5);
  return v9;
}
