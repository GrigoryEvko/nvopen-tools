// Function: sub_8A1CE0
// Address: 0x8a1ce0
//
const __m128i *__fastcall sub_8A1CE0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 *a5,
        int a6,
        __int64 ***a7,
        unsigned int a8,
        _DWORD *a9,
        __int64 a10)
{
  int v13; // eax
  unsigned int v14; // edx
  __int64 v15; // r12
  __int64 v16; // r11
  const __m128i *v17; // r8
  int v18; // eax
  __int64 v19; // r11
  int v20; // eax
  int v21; // eax
  _DWORD *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __int8 v25; // al
  __int64 v26; // r11
  char v27; // dl
  __int64 v28; // rdx
  int v29; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // edx
  char v34; // al
  __int64 v35; // rcx
  char v36; // dl
  __int64 v37; // rax
  __int64 v38; // rsi
  int v39; // eax
  int v40; // eax
  _BOOL4 v41; // eax
  __int64 v42; // [rsp+10h] [rbp-88h]
  __int64 v43; // [rsp+10h] [rbp-88h]
  __int64 v44; // [rsp+10h] [rbp-88h]
  __int64 v45; // [rsp+10h] [rbp-88h]
  const __m128i *v47; // [rsp+18h] [rbp-80h]
  const __m128i *v48; // [rsp+18h] [rbp-80h]
  _QWORD v49[2]; // [rsp+28h] [rbp-70h] BYREF
  __m128i v50; // [rsp+38h] [rbp-60h]
  __m128i v51; // [rsp+48h] [rbp-50h]
  __m128i v52; // [rsp+58h] [rbp-40h]
  int v53; // [rsp+B0h] [rbp+18h]

  v13 = *(_DWORD *)(a10 + 76);
  v14 = a8 & 0xFFFFFFFE;
  if ( v13 )
    v14 = a8;
  *(_DWORD *)(a10 + 76) = v13 + 1;
  v53 = v14;
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 256LL);
  if ( !v15 )
    v15 = a2;
  v16 = sub_8A2270(v15, a3, a4, (_DWORD)a5, v14 | 1, (_DWORD)a9, a10);
  if ( *a9 )
    goto LABEL_29;
  v17 = (const __m128i *)a1;
  if ( v15 == v16 )
    goto LABEL_26;
  v42 = v16;
  v18 = sub_8D3A70(v16);
  v19 = v42;
  if ( !v18 )
  {
    v20 = sub_8D3D40(v42);
    v19 = v42;
    if ( !v20 )
    {
      if ( !dword_4D044A0 )
        goto LABEL_28;
      v21 = sub_8D2870(v42);
      v19 = v42;
      if ( !v21 )
        goto LABEL_28;
    }
  }
  v22 = 0;
  if ( (_DWORD)qword_4F077B4 && qword_4F077A0 <= 0x1FBCFu )
    v22 = a9;
  if ( dword_4F077C4 == 2 )
  {
    v45 = v19;
    v39 = sub_8D23B0(v19);
    v19 = v45;
    if ( v39 )
    {
      v40 = sub_8D3A70(v45);
      v19 = v45;
      if ( v40 )
      {
        sub_8AD220(v45, v22);
        v19 = v45;
      }
    }
  }
  v43 = v19;
  if ( *a9 )
    goto LABEL_29;
  v23 = sub_894B00(a1);
  if ( v23 )
  {
    v24 = sub_8A2270(v23, a3, a4, (_DWORD)a5, v53, (_DWORD)a9, a10);
    if ( (unsigned __int8)(*(_BYTE *)(v43 + 140) - 9) > 2u )
      goto LABEL_28;
    v17 = sub_7D3640((const char *)v43, v24, (__int64)a5);
    goto LABEL_17;
  }
  v31 = *a5;
  v50 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v49[1] = v31;
  v51 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v32 = *(_QWORD *)a1;
  v52 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v49[0] = v32;
  if ( !(unsigned int)sub_8D2870(v43) )
  {
    v33 = 1;
    if ( (v53 & 1) != 0 )
    {
LABEL_43:
      sub_7D2AC0(v49, (const char *)v43, v33);
      v17 = (const __m128i *)v50.m128i_i64[1];
      if ( !v50.m128i_i64[1] )
        goto LABEL_26;
      if ( (*(_BYTE *)(v50.m128i_i64[1] + 82) & 4) != 0
        || (*(_BYTE *)(v50.m128i_i64[1] + 81) & 0x10) != 0
        && dword_4F07734
        && !dword_4F07730
        && (v48 = (const __m128i *)v50.m128i_i64[1], v41 = sub_884000(v50.m128i_i64[1], 1), v17 = v48, !v41) )
      {
        *a9 = 1;
      }
      goto LABEL_17;
    }
    v34 = *(_BYTE *)(a1 + 80);
    if ( !a6 )
    {
      v33 = (v34 == 19) << 13;
      goto LABEL_43;
    }
    v35 = *(_QWORD *)(a1 + 88);
    v36 = *(_BYTE *)(v35 + 140);
    if ( v34 == 3 )
    {
      if ( (unsigned __int8)(v36 - 9) <= 3u )
      {
        v37 = **(_QWORD **)(v35 + 168);
        goto LABEL_42;
      }
    }
    else if ( v34 == 6 )
    {
      if ( v36 == 12 || (unsigned __int8)(v36 - 9) <= 2u )
        goto LABEL_75;
    }
    else if ( v36 == 12 || (unsigned __int8)(v36 - 9) <= 2u )
    {
      if ( (unsigned __int8)(v34 - 4) <= 1u )
      {
        v37 = *(_QWORD *)(*(_QWORD *)(v35 + 168) + 168LL);
LABEL_42:
        v33 = v37 == 0 ? 1024 : 0x2000;
        goto LABEL_43;
      }
      if ( v34 == 7 )
      {
        v37 = **(_QWORD **)(v35 + 216);
        goto LABEL_42;
      }
LABEL_75:
      v37 = *(_QWORD *)(v35 + 240);
      goto LABEL_42;
    }
    v33 = 1024;
    goto LABEL_43;
  }
  v17 = (const __m128i *)sub_7D36A0((__int64)v49, v43);
LABEL_17:
  if ( !v17 )
  {
LABEL_29:
    v17 = 0;
    goto LABEL_26;
  }
  v25 = v17[5].m128i_i8[0];
  v26 = (__int64)v17;
  if ( v25 == 16 )
  {
    v26 = *(_QWORD *)v17[5].m128i_i64[1];
    v25 = *(_BYTE *)(v26 + 80);
  }
  if ( v25 == 24 )
  {
    v26 = *(_QWORD *)(v26 + 88);
    if ( !v26 )
      goto LABEL_26;
    v25 = *(_BYTE *)(v26 + 80);
  }
  v27 = *(_BYTE *)(a1 + 80);
  if ( v25 == 19 )
  {
    if ( v27 != 19 )
    {
      if ( (unsigned __int8)(v27 - 4) > 1u )
      {
        if ( v27 != 3 )
          goto LABEL_28;
        v38 = *(_QWORD *)(a1 + 88);
        if ( !v38 || *(_BYTE *)(v38 + 140) != 12 || *(_BYTE *)(v38 + 184) != 10 )
          goto LABEL_28;
LABEL_60:
        v17 = (const __m128i *)sub_8A1930(v26, (__int64 **)v38, a3, a4, (__int64)a5, v53, a9, a10, a7);
        goto LABEL_26;
      }
      v38 = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(v38 + 177) & 0x10) != 0 )
        goto LABEL_60;
LABEL_28:
      *a9 = 1;
      goto LABEL_29;
    }
  }
  else if ( (unsigned __int8)(v27 - 4) <= 1u )
  {
    v28 = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(v28 + 177) & 0x30) == 0x30 )
    {
      if ( *(_QWORD *)(*(_QWORD *)(v28 + 168) + 168LL) )
        goto LABEL_28;
    }
  }
  v47 = v17;
  v44 = v26;
  v29 = sub_8D3D40(v15);
  v17 = v47;
  if ( v29 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 80) - 4) <= 1u )
    {
      v38 = *(_QWORD *)(a1 + 88);
      v26 = v44;
      if ( (*(_BYTE *)(v38 + 177) & 0x30) == 0x30 )
      {
        if ( *(_BYTE *)(v44 + 80) == 19 )
          goto LABEL_60;
        goto LABEL_28;
      }
    }
  }
LABEL_26:
  --*(_DWORD *)(a10 + 76);
  return v17;
}
