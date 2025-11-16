// Function: sub_234FBC0
// Address: 0x234fbc0
//
__int64 __fastcall sub_234FBC0(__int64 a1, __int64 a2, __int64 a3)
{
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __int32 v6; // r15d
  unsigned int v7; // eax
  unsigned int v8; // ebx
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned int v11; // edx
  __int64 v12; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  char *v17; // rdx
  __int64 v18; // rax
  char v19; // si
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rsi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdx
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // [rsp+8h] [rbp-148h]
  unsigned int v36; // [rsp+18h] [rbp-138h]
  unsigned int v37; // [rsp+18h] [rbp-138h]
  __m128i v38; // [rsp+20h] [rbp-130h] BYREF
  __int64 v39; // [rsp+38h] [rbp-118h] BYREF
  __m128i v40; // [rsp+40h] [rbp-110h] BYREF
  __m128i v41; // [rsp+50h] [rbp-100h] BYREF
  __m128i v42; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v43; // [rsp+70h] [rbp-E0h] BYREF
  unsigned __int64 v44; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v45; // [rsp+80h] [rbp-D0h] BYREF
  char *v46; // [rsp+88h] [rbp-C8h]
  __int64 v47; // [rsp+90h] [rbp-C0h]
  unsigned __int64 v48[4]; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v50; // [rsp+D0h] [rbp-80h] BYREF
  char v51; // [rsp+E0h] [rbp-70h]
  void *v52; // [rsp+E8h] [rbp-68h] BYREF
  __m128i *v53; // [rsp+F0h] [rbp-60h]
  void **v54; // [rsp+F8h] [rbp-58h] BYREF
  __m128i *v55; // [rsp+100h] [rbp-50h]
  void ***v56; // [rsp+108h] [rbp-48h] BYREF
  void **v57; // [rsp+110h] [rbp-40h]

  v38.m128i_i64[0] = a2;
  v38.m128i_i64[1] = a3;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  if ( !a3 )
  {
    v18 = 0;
    v17 = 0;
    v16 = 0;
LABEL_18:
    v19 = *(_BYTE *)(a1 + 24);
    *(_QWORD *)a1 = v16;
    *(_QWORD *)(a1 + 8) = v17;
    *(_QWORD *)(a1 + 16) = v18;
    *(_BYTE *)(a1 + 24) = v19 & 0xFC | 2;
    return a1;
  }
  while ( 2 )
  {
    v40 = 0u;
    LOBYTE(v48[0]) = 59;
    sub_232E160(&v49, &v38, v48, 1u);
    v4 = _mm_loadu_si128(&v50);
    v40 = _mm_loadu_si128(&v49);
    v38 = v4;
    if ( v40.m128i_i64[1] <= 7uLL || *(_QWORD *)v40.m128i_i64[0] != 0x5B7366666F747563LL )
    {
      v14 = sub_C63BB0();
      v50.m128i_i64[1] = 1;
      v8 = v14;
      v51 = 1;
      v10 = v15;
      v49.m128i_i64[0] = (__int64)"invalid LowerAllowCheck pass parameter '{0}' ";
      v50.m128i_i64[0] = (__int64)&v54;
      v49.m128i_i64[1] = 45;
      v52 = &unk_49DB108;
      v53 = &v40;
      v54 = &v52;
      goto LABEL_7;
    }
    v41 = 0u;
    v42 = 0u;
    sub_232E160(&v49, &v40, "]=", 2u);
    v5 = _mm_loadu_si128(&v49);
    v42 = v50;
    v41 = v5;
    if ( sub_C93CC0(v50.m128i_i64[0], v50.m128i_i64[1], 0, v49.m128i_i64)
      || (v6 = v49.m128i_i32[0], v49.m128i_i64[0] != v49.m128i_i32[0]) )
    {
      v7 = sub_C63BB0();
      v51 = 1;
      v8 = v7;
      v50.m128i_i64[1] = 2;
      v10 = v9;
      v49.m128i_i64[0] = (__int64)"invalid LowerAllowCheck pass cutoffs parameter '{0}' ({1})";
      v50.m128i_i64[0] = (__int64)&v56;
      v49.m128i_i64[1] = 58;
      v53 = &v38;
      v52 = &unk_49DB108;
      v54 = (void **)&unk_49DB108;
      v55 = &v42;
      v56 = &v54;
      v57 = &v52;
LABEL_7:
      sub_23328D0((__int64)v48, (__int64)&v49);
      v11 = v8;
      goto LABEL_8;
    }
    if ( (unsigned __int8)sub_95CB50((const void **)&v41, "cutoffs[", 8u)
      && !sub_9691B0((const void *)v41.m128i_i64[0], v41.m128i_u64[1], byte_3F871B3, 0) )
    {
      while ( v41.m128i_i64[1] )
      {
        v43 = 0;
        v44 = 0;
        v49.m128i_i8[0] = 124;
        v20 = sub_C931B0(v41.m128i_i64, &v49, 1u, 0);
        if ( v20 == -1 )
        {
          v20 = v41.m128i_u64[1];
          v22 = v41.m128i_i64[0];
          v23 = 0;
          v24 = 0;
        }
        else
        {
          v21 = v20 + 1;
          v22 = v41.m128i_i64[0];
          if ( v20 + 1 > v41.m128i_i64[1] )
          {
            v21 = v41.m128i_u64[1];
            v23 = 0;
          }
          else
          {
            v23 = v41.m128i_i64[1] - v21;
          }
          v24 = v41.m128i_i64[0] + v21;
          if ( v20 > v41.m128i_i64[1] )
            v20 = v41.m128i_u64[1];
        }
        v43 = v22;
        v41.m128i_i64[1] = v23;
        v41.m128i_i64[0] = v24;
        v44 = v20;
        if ( sub_C93C90(v22, v20, 0, (unsigned __int64 *)&v49)
          || (v25 = v49.m128i_u32[0], v49.m128i_i64[0] != v49.m128i_u32[0]) )
        {
          v26 = sub_C63BB0();
          v51 = 1;
          v28 = v27;
          v53 = &v41;
          v49.m128i_i64[0] = (__int64)"invalid LowerAllowCheck pass index parameter '{0}' ({1}) {2}";
          v50.m128i_i64[0] = (__int64)&v56;
          v36 = v26;
          v50.m128i_i64[1] = 2;
          v52 = &unk_49DB108;
          v54 = (void **)&unk_49DB108;
          v55 = (__m128i *)&v43;
          v56 = &v54;
          v57 = &v52;
          v49.m128i_i64[1] = 60;
          sub_23328D0((__int64)v48, (__int64)&v49);
          sub_23058C0(&v39, (__int64)v48, v36, v28);
          v29 = v39;
          *(_BYTE *)(a1 + 24) |= 3u;
          *(_QWORD *)a1 = v29 & 0xFFFFFFFFFFFFFFFELL;
          sub_2240A30(v48);
          goto LABEL_9;
        }
        v30 = v45;
        v31 = (__int64)&v46[-v45] >> 2;
        if ( v31 <= v49.m128i_u32[0] )
        {
          v49.m128i_i32[0] = 0;
          v32 = (unsigned int)(v25 + 1);
          if ( v32 > v31 )
          {
            v35 = v25;
            sub_1CFD340((__int64)&v45, v46, v32 - v31, &v49);
            v30 = v45;
            v25 = v35;
          }
          else if ( v32 < v31 && v46 != (char *)(v45 + 4 * v32) )
          {
            v46 = (char *)(v45 + 4 * v32);
          }
        }
        *(_DWORD *)(v30 + 4 * v25) = v6;
      }
      if ( v38.m128i_i64[1] )
        continue;
      v16 = v45;
      v17 = v46;
      v18 = v47;
      goto LABEL_18;
    }
    break;
  }
  v33 = sub_C63BB0();
  v51 = 1;
  v10 = v34;
  v55 = &v41;
  v49.m128i_i64[0] = (__int64)"invalid LowerAllowCheck pass index parameter '{0}' ({1})";
  v50.m128i_i64[0] = (__int64)&v56;
  v37 = v33;
  v50.m128i_i64[1] = 2;
  v52 = &unk_49DB108;
  v54 = (void **)&unk_49DB108;
  v56 = &v54;
  v49.m128i_i64[1] = 56;
  v53 = &v42;
  v57 = &v52;
  sub_23328D0((__int64)v48, (__int64)&v49);
  v11 = v37;
LABEL_8:
  sub_23058C0(&v43, (__int64)v48, v11, v10);
  v12 = v43;
  *(_BYTE *)(a1 + 24) |= 3u;
  *(_QWORD *)a1 = v12 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v48);
LABEL_9:
  if ( v45 )
    j_j___libc_free_0(v45);
  return a1;
}
