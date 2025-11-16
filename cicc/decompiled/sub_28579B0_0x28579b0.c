// Function: sub_28579B0
// Address: 0x28579b0
//
__int64 __fastcall sub_28579B0(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // r14
  __int16 v5; // ax
  __int64 v6; // r15
  unsigned int v7; // esi
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v12; // rdi
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // rax
  const void *v19; // r9
  size_t v20; // r14
  __int64 v21; // r8
  _BYTE *v22; // rdi
  __m128i v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rax
  __int64 v26; // r8
  const void *v27; // r10
  __int64 v28; // r8
  __int64 v29; // r9
  _BYTE *v30; // rdi
  __m128i v31; // rax
  __int64 *v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-C8h]
  const void *v37; // [rsp+10h] [rbp-C0h]
  const void *v38; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v39; // [rsp+18h] [rbp-B8h]
  int v40; // [rsp+18h] [rbp-B8h]
  __int64 v41; // [rsp+18h] [rbp-B8h]
  int v42; // [rsp+18h] [rbp-B8h]
  __m128i v43; // [rsp+30h] [rbp-A0h] BYREF
  __m128i v44; // [rsp+40h] [rbp-90h]
  _BYTE *v45; // [rsp+50h] [rbp-80h] BYREF
  __int64 v46; // [rsp+58h] [rbp-78h]
  _BYTE v47[112]; // [rsp+60h] [rbp-70h] BYREF

  v4 = *(_QWORD **)a1;
  v5 = *(_WORD *)(*(_QWORD *)a1 + 24LL);
  if ( !v5 )
  {
    v6 = v4[4];
    v7 = *(_DWORD *)(v6 + 32);
    v8 = 1LL << ((unsigned __int8)v7 - 1);
    v9 = *(_QWORD *)(v6 + 24);
    if ( v7 > 0x40 )
    {
      v12 = v6 + 24;
      v13 = v7 + 1;
      if ( (*(_QWORD *)(v9 + 8LL * ((v7 - 1) >> 6)) & v8) != 0 )
      {
        if ( v13 - (unsigned int)sub_C44500(v12) > 0x40 )
          goto LABEL_7;
        goto LABEL_11;
      }
      v10 = v13 - sub_C444A0(v12);
    }
    else
    {
      if ( (v8 & v9) == 0 )
      {
        if ( v9 )
        {
          _BitScanReverse64(&v9, v9);
          v10 = 65 - (v9 ^ 0x3F);
          goto LABEL_6;
        }
LABEL_11:
        *(_QWORD *)a1 = sub_DA2C50((__int64)a2, *(_QWORD *)(v6 + 8), 0, 0);
        v14 = v4[4];
        v15 = *(__int64 **)(v14 + 24);
        v16 = *(_DWORD *)(v14 + 32);
        if ( v16 > 0x40 )
        {
          v17 = *v15;
        }
        else
        {
          v17 = 0;
          if ( v16 )
            v17 = (__int64)((_QWORD)v15 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
        }
        v44.m128i_i64[0] = v17;
        v44.m128i_i8[8] = 0;
        return v44.m128i_i64[0];
      }
      if ( !v7 )
        goto LABEL_11;
      v25 = ~(v9 << (64 - (unsigned __int8)v7));
      if ( v25 )
      {
        _BitScanReverse64(&v25, v25);
        v10 = v7 + 1 - (v25 ^ 0x3F);
      }
      else
      {
        v10 = v7 - 63;
      }
    }
LABEL_6:
    if ( v10 > 0x40 )
    {
LABEL_7:
      v44.m128i_i64[0] = 0;
      v44.m128i_i8[8] = 0;
      return v44.m128i_i64[0];
    }
    goto LABEL_11;
  }
  if ( v5 != 5 )
  {
    if ( v5 != 8 )
    {
      if ( v5 == 6 )
      {
        if ( (_BYTE)qword_5001068 )
        {
          if ( v4[5] == 2 )
          {
            v32 = (__int64 *)v4[4];
            v33 = *v32;
            if ( !*(_WORD *)(*v32 + 24) && *(_WORD *)(v32[1] + 24) == 1 )
            {
              v34 = sub_D95540(v33);
              *(_QWORD *)a1 = sub_DA2C50((__int64)a2, v34, 0, 0);
              v35 = sub_2850C10(*(_QWORD *)(*(_QWORD *)(v33 + 32) + 24LL), *(_DWORD *)(*(_QWORD *)(v33 + 32) + 32LL));
              v44.m128i_i8[8] = 1;
              v44.m128i_i64[0] = v35;
              return v44.m128i_i64[0];
            }
          }
        }
      }
      goto LABEL_7;
    }
    v26 = v4[5];
    v27 = (const void *)v4[4];
    v45 = v47;
    v28 = 8 * v26;
    v46 = 0x800000000LL;
    v29 = v28 >> 3;
    if ( (unsigned __int64)v28 > 0x40 )
    {
      v36 = v28;
      v38 = v27;
      v41 = v28 >> 3;
      sub_C8D5F0((__int64)&v45, v47, v28 >> 3, 8u, v28, v29);
      LODWORD(v29) = v41;
      v27 = v38;
      v28 = v36;
      v30 = &v45[8 * (unsigned int)v46];
    }
    else
    {
      v30 = v47;
      if ( !v28 )
        goto LABEL_29;
    }
    v42 = v29;
    memcpy(v30, v27, v28);
    v30 = v45;
    LODWORD(v28) = v46;
    LODWORD(v29) = v42;
LABEL_29:
    LODWORD(v46) = v29 + v28;
    v31.m128i_i64[0] = sub_28579B0(v30, a2);
    v43 = v31;
    if ( v31.m128i_i64[0] )
      *(_QWORD *)a1 = sub_DBFF60((__int64)a2, (unsigned int *)&v45, v4[6], 0);
    v24 = (unsigned __int64)v45;
    v44 = _mm_loadu_si128(&v43);
    if ( v45 != v47 )
      goto LABEL_21;
    return v44.m128i_i64[0];
  }
  v18 = v4[5];
  v19 = (const void *)v4[4];
  v45 = v47;
  v20 = 8 * v18;
  v46 = 0x800000000LL;
  v21 = (8 * v18) >> 3;
  if ( (unsigned __int64)(8 * v18) > 0x40 )
  {
    v37 = v19;
    v39 = (8 * v18) >> 3;
    sub_C8D5F0((__int64)&v45, v47, v39, 8u, v21, (__int64)v19);
    LODWORD(v21) = v39;
    v19 = v37;
    v22 = &v45[8 * (unsigned int)v46];
  }
  else
  {
    v22 = v47;
    if ( !v20 )
      goto LABEL_18;
  }
  v40 = v21;
  memcpy(v22, v19, v20);
  v22 = v45;
  LODWORD(v20) = v46;
  LODWORD(v21) = v40;
LABEL_18:
  LODWORD(v46) = v21 + v20;
  v23.m128i_i64[0] = sub_28579B0(v22, a2);
  v43 = v23;
  if ( v23.m128i_i64[0] )
    *(_QWORD *)a1 = sub_DC7EB0(a2, (__int64)&v45, 0, 0);
  v24 = (unsigned __int64)v45;
  v44 = _mm_loadu_si128(&v43);
  if ( v45 != v47 )
LABEL_21:
    _libc_free(v24);
  return v44.m128i_i64[0];
}
