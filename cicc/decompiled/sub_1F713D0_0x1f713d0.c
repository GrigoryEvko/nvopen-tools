// Function: sub_1F713D0
// Address: 0x1f713d0
//
__int64 __fastcall sub_1F713D0(__int64 a1, __int64 a2, __m128i *a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int16 i; // r12
  _DWORD *v10; // rdx
  __int64 v11; // rdi
  char v12; // r15
  bool v13; // al
  __int8 v14; // r10
  __int64 v15; // rsi
  char v16; // cl
  char v17; // cl
  char v18; // r10
  int v19; // r15d
  int v20; // eax
  bool v21; // al
  __int64 v22; // rdx
  __int16 v23; // ax
  __int64 v24; // r10
  __int64 v25; // rax
  char v26; // r15
  __int64 v27; // rax
  char v28; // di
  int v29; // eax
  __m128i v30; // xmm0
  __m128i v31; // xmm1
  __int8 v32; // al
  int v34; // eax
  int v35; // r12d
  int v36; // eax
  char v37; // [rsp+0h] [rbp-90h]
  char v38; // [rsp+0h] [rbp-90h]
  char v39; // [rsp+Bh] [rbp-85h]
  unsigned int v40; // [rsp+Ch] [rbp-84h]
  char v43[8]; // [rsp+28h] [rbp-68h] BYREF
  __m128i v44; // [rsp+30h] [rbp-60h] BYREF
  __m128i v45; // [rsp+40h] [rbp-50h] BYREF
  __int64 v46; // [rsp+50h] [rbp-40h]
  __int8 v47; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(v6 + 40);
  v8 = *(_QWORD *)(v6 + 48);
  for ( i = *(_WORD *)(v7 + 24); i == 158; LODWORD(v8) = v10[2] )
  {
    v10 = *(_DWORD **)(v7 + 32);
    v7 = *(_QWORD *)v10;
    i = *(_WORD *)(*(_QWORD *)v10 + 24LL);
  }
  v11 = *(_QWORD *)a1;
  v40 = v8;
  v12 = *(_BYTE *)v11;
  if ( *(_BYTE *)v11 )
    v13 = (unsigned __int8)(v12 - 14) <= 0x47u || (unsigned __int8)(v12 - 2) <= 5u;
  else
    v13 = sub_1F58CF0(v11);
  v14 = *(_BYTE *)(a2 + 88);
  v15 = *(_QWORD *)(a2 + 96);
  v16 = **(_BYTE **)(a1 + 8);
  if ( v13 )
  {
    v44.m128i_i8[0] = v14;
    v44.m128i_i64[1] = v15;
    if ( v12 != v14 )
    {
      if ( v12 )
      {
        v19 = sub_1F6C8D0(v12);
        if ( v18 )
        {
LABEL_9:
          v20 = sub_1F6C8D0(v18);
LABEL_10:
          v21 = v20 != v19;
LABEL_11:
          if ( !v16 )
          {
            if ( !**(_BYTE **)(a1 + 40) )
              goto LABEL_23;
            if ( v21 )
              return 0;
            goto LABEL_21;
          }
          if ( v21 || i != 185 )
            return 0;
          goto LABEL_14;
        }
LABEL_41:
        v38 = v17;
        v20 = sub_1F58D40((__int64)&v44);
        v16 = v38;
        goto LABEL_10;
      }
LABEL_40:
      v39 = v14;
      v37 = v16;
      v34 = sub_1F58D40(v11);
      v18 = v39;
      v17 = v37;
      v19 = v34;
      if ( v39 )
        goto LABEL_9;
      goto LABEL_41;
    }
    if ( !v12 && v15 != *(_QWORD *)(v11 + 8) )
      goto LABEL_40;
  }
  else
  {
    if ( v12 != v14 )
    {
      if ( v16 || **(_BYTE **)(a1 + 40) )
        return 0;
      goto LABEL_23;
    }
    if ( !v12 )
    {
      v21 = *(_QWORD *)(v11 + 8) != v15;
      goto LABEL_11;
    }
  }
  if ( !v16 )
    goto LABEL_20;
  if ( i != 185 )
    return 0;
LABEL_14:
  sub_2043720(&v44, v7, **(_QWORD **)(a1 + 16));
  v22 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v7 + 88) != *(_BYTE *)v22
    || !*(_BYTE *)v22 && *(_QWORD *)(v7 + 96) != *(_QWORD *)(v22 + 8)
    || !sub_1D18C00(v7, 1, 0)
    || (*(_BYTE *)(v7 + 26) & 8) != 0
    || (*(_WORD *)(v7 + 26) & 0x380) != 0
    || !(unsigned __int8)sub_2043540(*(_QWORD *)(a1 + 32), &v44, **(_QWORD **)(a1 + 16), v43) )
  {
    return 0;
  }
LABEL_20:
  if ( !**(_BYTE **)(a1 + 40) )
    goto LABEL_23;
LABEL_21:
  v23 = *(_WORD *)(v7 + 24);
  if ( (unsigned __int16)(v23 - 10) > 1u && (unsigned __int16)(v23 - 32) > 1u )
    return 0;
LABEL_23:
  if ( !**(_BYTE **)(a1 + 48) )
  {
LABEL_29:
    sub_2043720(&v44, a2, **(_QWORD **)(a1 + 16));
    v30 = _mm_loadu_si128(&v44);
    v31 = _mm_loadu_si128(&v45);
    a3[2].m128i_i64[0] = v46;
    v32 = v47;
    *a3 = v30;
    a3[2].m128i_i8[8] = v32;
    a3[1] = v31;
    return sub_2043540(*(_QWORD *)(a1 + 56), a3, **(_QWORD **)(a1 + 16), a4);
  }
  if ( (*(_BYTE *)(a2 + 27) & 4) == 0 )
  {
    v24 = *(_QWORD *)a1;
    v25 = *(_QWORD *)(v7 + 40) + 16LL * v40;
    v26 = *(_BYTE *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    v44.m128i_i8[0] = v26;
    v44.m128i_i64[1] = v27;
    v28 = *(_BYTE *)v24;
    if ( v26 == *(_BYTE *)v24 )
    {
      if ( v26 || v27 == *(_QWORD *)(v24 + 8) )
      {
LABEL_27:
        v29 = *(unsigned __int16 *)(v7 + 24);
        if ( v29 == 106 || v29 == 109 )
          goto LABEL_29;
        return 0;
      }
    }
    else if ( v28 )
    {
      v35 = sub_1F6C8D0(v28);
      goto LABEL_50;
    }
    v35 = sub_1F58D40(v24);
LABEL_50:
    if ( v26 )
      v36 = sub_1F6C8D0(v26);
    else
      v36 = sub_1F58D40((__int64)&v44);
    if ( v36 != v35 )
      return 0;
    goto LABEL_27;
  }
  return 0;
}
