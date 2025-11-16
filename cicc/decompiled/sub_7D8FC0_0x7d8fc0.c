// Function: sub_7D8FC0
// Address: 0x7d8fc0
//
_QWORD *__fastcall sub_7D8FC0(__int64 a1)
{
  __int64 v2; // r14
  unsigned __int8 v3; // bl
  _QWORD *i; // r15
  unsigned __int8 v5; // r13
  _QWORD *v6; // rbx
  _QWORD *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rax
  _QWORD *v13; // r9
  __int64 v14; // rax
  char **v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rax
  const __m128i *v22; // r13
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // rax
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  _BYTE *v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int128 *v32; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+28h] [rbp-58h]
  int v34; // [rsp+34h] [rbp-4Ch]
  char **v35; // [rsp+38h] [rbp-48h]
  int v36; // [rsp+44h] [rbp-3Ch] BYREF
  const __m128i *v37; // [rsp+48h] [rbp-38h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = *(_BYTE *)(a1 + 56);
  for ( i = *(_QWORD **)(a1 + 72); *(_BYTE *)(v2 + 140) == 12; v2 = *(_QWORD *)(v2 + 160) )
    ;
  v37 = (const __m128i *)sub_724DC0();
  if ( v3 == 37 )
  {
    v32 = &xmmword_4F18700;
    v35 = &off_4B7B2A0;
  }
  else
  {
    if ( v3 <= 0x25u )
    {
      if ( v3 == 35 )
      {
        v5 = *(_BYTE *)(v2 + 160);
        sub_72C890(v5, "1.0", "0.0", (__int64)v37);
        v6 = sub_73A720(v37, (__int64)"1.0");
        *(_BYTE *)(v6[7] - 8LL) &= ~8u;
        sub_7D8EC0(v6);
        v32 = &xmmword_4F18700;
        v35 = &off_4B7B2A0;
        goto LABEL_8;
      }
      if ( v3 == 36 )
      {
        v5 = *(_BYTE *)(v2 + 160);
        sub_72C890(v5, "1.0", "0.0", (__int64)v37);
        v6 = sub_73A720(v37, (__int64)"1.0");
        *(_BYTE *)(v6[7] - 8LL) &= ~8u;
        sub_7D8EC0(v6);
        v32 = &xmmword_4F186A0;
        v35 = &off_4B7B240;
LABEL_8:
        v34 = ((*(_BYTE *)(a1 + 25) >> 2) ^ 1) & 1;
        goto LABEL_9;
      }
LABEL_32:
      sub_721090();
    }
    if ( v3 != 38 )
      goto LABEL_32;
    v32 = &xmmword_4F186A0;
    v35 = &off_4B7B240;
  }
  v5 = *(_BYTE *)(v2 + 160);
  sub_72C890(v5, "1.0", "0.0", (__int64)v37);
  v6 = sub_73A720(v37, (__int64)"1.0");
  *(_BYTE *)(v6[7] - 8LL) &= ~8u;
  sub_7D8EC0(v6);
  v34 = 0;
LABEL_9:
  v7 = 0;
  v33 = 0;
  v11 = sub_7E2590(i, 0, &v36);
  v31 = 0;
  if ( v36 | v34 )
  {
    v29 = (_QWORD *)v11;
    v33 = sub_7E7CB0(v2);
    v30 = sub_731250(v33);
    v7 = sub_731370((__int64)i, 0, v24, v25, v26, v27);
    v31 = sub_7E2BE0(v33, v7);
    i = v29;
    v11 = (__int64)v30;
  }
  v12 = sub_731370(v11, (__int64)v7, v8, v9, v10, v11);
  v12[2] = v6;
  v13 = v12;
  v14 = 8LL * v5;
  if ( v5 != 10 )
  {
    if ( v5 == 11 )
    {
      v35 += 2;
    }
    else if ( v5 == 12 )
    {
      v35 += 4;
    }
    else
    {
      v15 = v35 + 8;
      if ( v5 != 13 )
        v15 = &v35[(unsigned __int64)v14 / 8];
      v35 = v15;
    }
  }
  v16 = *(_QWORD *)((char *)v32 + v14);
  if ( v16 )
    v17 = sub_7F88E0(v16, v13);
  else
    v17 = sub_7F8B20(*v35, (char *)v32 + v14, v2, v2, v2, v13);
  v21 = (__int64 *)sub_698020(i, 73, v17, v18, v19, v20);
  v22 = (const __m128i *)v21;
  if ( v33 )
    v22 = (const __m128i *)sub_73DF90(v31, v21);
  if ( v34 )
  {
    v28 = sub_73E830(v33);
    v22 = (const __m128i *)sub_73DF90((__int64)v22, v28);
  }
  sub_730620(a1, v22);
  return sub_724E30((__int64)&v37);
}
