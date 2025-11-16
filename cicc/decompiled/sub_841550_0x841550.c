// Function: sub_841550
// Address: 0x841550
//
_BOOL8 __fastcall sub_841550(__m128i *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 **v4; // rax
  __int64 v6; // r13
  __m128i *v7; // r14
  bool v8; // zf
  __int64 v10; // rcx
  char v11; // al
  __m128i *v12; // r10
  __m128i *i; // r8
  char v14; // al
  __int64 v15; // rcx
  _BOOL4 v16; // r9d
  unsigned int v17; // eax
  int v18; // eax
  __int64 *k; // r13
  __int64 *v20; // rbx
  __int64 v22; // rax
  char v23; // al
  char v24; // al
  int v25; // eax
  int v26; // eax
  __int64 v27; // rcx
  __int64 j; // rdx
  char v29; // al
  _QWORD *v30; // r13
  __m128i *v31; // [rsp+8h] [rbp-78h]
  __m128i *v32; // [rsp+10h] [rbp-70h]
  const __m128i *v33; // [rsp+18h] [rbp-68h]
  _BOOL4 v34; // [rsp+18h] [rbp-68h]
  const __m128i **v35; // [rsp+20h] [rbp-60h]
  _BOOL4 v36; // [rsp+20h] [rbp-60h]
  _BOOL4 v37; // [rsp+20h] [rbp-60h]
  _BOOL4 v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-60h]
  __int64 v40; // [rsp+20h] [rbp-60h]
  unsigned int v41; // [rsp+30h] [rbp-50h] BYREF
  int v42; // [rsp+34h] [rbp-4Ch] BYREF
  int v43; // [rsp+38h] [rbp-48h] BYREF
  int v44; // [rsp+3Ch] [rbp-44h] BYREF
  int v45; // [rsp+40h] [rbp-40h] BYREF
  int v46; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 *v47; // [rsp+48h] [rbp-38h] BYREF

  v4 = 0;
  v6 = a1->m128i_i64[0];
  v7 = *(__m128i **)a2;
  if ( !a4 )
    v4 = &v47;
  *(_OWORD *)a3 = 0;
  *(_OWORD *)(a3 + 16) = 0;
  *(_OWORD *)(a3 + 32) = 0;
  v8 = *(_BYTE *)(a2 + 16) == 3;
  v41 = 0;
  v47 = 0;
  v35 = (const __m128i **)v4;
  if ( v8 )
    return 0;
  if ( *(_BYTE *)(a2 + 17) != 1 || sub_6ED0A0(a2) )
  {
    v33 = 0;
  }
  else
  {
    v33 = (const __m128i *)sub_72D600(v7);
    if ( a1[1].m128i_i8[1] != 2
      && !sub_6ED0A0((__int64)a1)
      && (unsigned int)sub_831CF0((__int64)a1, 0, (__int64)v33, 0, 0, &v42, &v43, &v44, &v45, &v46, 0) )
    {
      *(_BYTE *)(a3 + 16) |= 0x14u;
      v17 = v41;
      v16 = 1;
      if ( !a4 )
      {
LABEL_51:
        if ( !v17 )
          return 1;
        goto LABEL_19;
      }
LABEL_29:
      *a4 = v17;
      return v16;
    }
    if ( (*(_BYTE *)(qword_4D03C50 + 16LL) > 3u || word_4D04898) && (unsigned int)sub_8E31E0(v6) )
    {
      v8 = (unsigned int)sub_8413E0(a1, (__int64)v33, 0, 1, a3, &v41, v35) == 0;
      v17 = v41;
      if ( !v8 )
        goto LABEL_44;
      if ( v41 )
        goto LABEL_28;
    }
  }
  if ( !(unsigned int)sub_8D3A70(v6) || !(unsigned int)sub_8D3A70(v7) )
  {
LABEL_24:
    v33 = sub_73D720(v7);
    if ( (unsigned int)sub_8D3A70(v7) )
    {
      v8 = (unsigned int)sub_836C50(a1, 0, v33, 1u, 1u, 1u, 0, 0, 0, a3, 0, &v41, (__int64 **)v35) == 0;
      v17 = v41;
      if ( v8 )
        goto LABEL_27;
    }
    else
    {
      if ( !(unsigned int)sub_8D3A70(v6) )
        sub_721090();
      v8 = (unsigned int)sub_840360(a1->m128i_i64, (__int64)v33, 0, 0, 1, 1, 0, 0, 0, a3, &v41, v35) == 0;
      v17 = v41;
      if ( v8 )
      {
LABEL_27:
        if ( !v17 )
        {
          v16 = 0;
          if ( a4 )
          {
            *a4 = 0;
            return v16;
          }
          return 0;
        }
LABEL_28:
        v16 = 1;
        if ( a4 )
          goto LABEL_29;
        goto LABEL_19;
      }
    }
LABEL_44:
    v16 = 1;
    if ( a4 )
    {
      *a4 = v17;
      return v16;
    }
    goto LABEL_51;
  }
  v11 = *(_BYTE *)(v6 + 140);
  v12 = (__m128i *)v6;
  if ( v11 == 12 )
  {
    do
      v12 = (__m128i *)v12[10].m128i_i64[0];
    while ( v12[8].m128i_i8[12] == 12 );
  }
  for ( i = v7; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  if ( v12 == i )
    goto LABEL_15;
  v31 = i;
  v32 = v12;
  if ( (unsigned int)sub_8D97D0(v12, i, 0, v10, i) )
  {
    v11 = *(_BYTE *)(v6 + 140);
LABEL_15:
    if ( (v11 & 0xFB) == 8 )
    {
      v16 = sub_8D5780(v7, v6) == 0;
      v14 = 0;
      v15 = 0;
    }
    else
    {
      v14 = 0;
      v15 = 0;
      v16 = 1;
    }
    goto LABEL_17;
  }
  v22 = sub_8D5CE0(v32, v31);
  v15 = v22;
  if ( !v22 )
  {
    if ( sub_8D5CE0(v31, v32) )
    {
      v17 = v41;
      v16 = 0;
      if ( a4 )
        goto LABEL_29;
      if ( v41 )
        goto LABEL_19;
      return 0;
    }
    goto LABEL_24;
  }
  if ( (*(_BYTE *)(v6 + 140) & 0xFB) == 8 )
  {
    v39 = v22;
    v25 = sub_8D5780(v7, v6);
    v15 = v39;
    v16 = v25 == 0;
  }
  else
  {
    v16 = 1;
  }
  if ( (*(_BYTE *)(v15 + 96) & 4) != 0 )
  {
    v23 = *(_BYTE *)(a3 + 16);
    v41 = 1;
    v24 = v23 | 8;
    *(_BYTE *)(a3 + 16) = v24;
    if ( a4 )
    {
      *(_BYTE *)(a3 + 36) |= 0x20u;
      *(_QWORD *)(a3 + 24) = v15;
      *(_BYTE *)(a3 + 16) = v24 & 0xEB | 0x10;
      *a4 = 1;
      return v16;
    }
    v34 = v16;
    v40 = v15;
    v26 = sub_6E5430();
    v27 = v40;
    v16 = v34;
    if ( v26 )
    {
      for ( j = *(_QWORD *)a2; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      sub_685360(0x11Eu, &a1[4].m128i_i32[1], j);
      v27 = v40;
      v16 = v34;
    }
    v29 = *(_BYTE *)(a3 + 16);
    *(_BYTE *)(a3 + 36) |= 0x20u;
    *(_QWORD *)(a3 + 24) = v27;
    *(_BYTE *)(a3 + 16) = v29 & 0xEB | 0x10;
    if ( v41 )
      goto LABEL_23;
    return v16;
  }
  v14 = 1;
LABEL_17:
  *(_QWORD *)(a3 + 24) = v15;
  *(_BYTE *)(a3 + 36) = (32 * v14) | *(_BYTE *)(a3 + 36) & 0xDF;
  *(_BYTE *)(a3 + 16) = *(_BYTE *)(a3 + 16) & 0xEB | 0x10;
  v17 = v41;
  if ( a4 )
    goto LABEL_29;
  if ( v41 )
  {
LABEL_19:
    v36 = v16;
    v18 = sub_6E5430();
    v16 = v36;
    if ( v18 )
    {
      v30 = sub_67DAA0(0x15Cu, &a1[4].m128i_i32[1], a1->m128i_i64[0], (__int64)v33);
      sub_82E650(v47, 0, 0, 0, v30);
      sub_685910((__int64)v30, 0);
      v16 = v36;
    }
    for ( k = v47; k; qword_4D03C68 = v20 )
    {
      v20 = k;
      v37 = v16;
      k = (__int64 *)*k;
      sub_725130((__int64 *)v20[5]);
      sub_82D8A0((_QWORD *)v20[15]);
      v16 = v37;
      *v20 = (__int64)qword_4D03C68;
    }
LABEL_23:
    v38 = v16;
    sub_6E6840((__int64)a1);
    return v38;
  }
  return v16;
}
