// Function: sub_31A8570
// Address: 0x31a8570
//
void __fastcall sub_31A8570(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  int v17; // ecx
  unsigned __int64 *v18; // r15
  unsigned __int64 *v19; // r13
  unsigned __int64 *v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  unsigned __int64 *v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 *v31; // r13
  __int64 v32; // r8
  unsigned __int64 v33; // rdi
  __int64 v34; // rsi
  __m128i *v35; // rdx
  unsigned __int64 *v36; // rdi
  const __m128i *v37; // rax
  unsigned __int64 *v38; // rsi
  unsigned __int64 v39; // r8
  const __m128i *v40; // rdi
  __m128i v41; // xmm6
  __int64 v42; // rdi
  __m128i *v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rcx
  __int64 v52; // r8
  unsigned __int64 v53; // rdi
  __int32 v54; // eax
  int v55; // [rsp+0h] [rbp-3C0h]
  unsigned __int64 *v56; // [rsp+0h] [rbp-3C0h]
  __int64 v57; // [rsp+8h] [rbp-3B8h]
  __int64 v58; // [rsp+8h] [rbp-3B8h]
  __int64 v59; // [rsp+18h] [rbp-3A8h] BYREF
  __m128i v60; // [rsp+20h] [rbp-3A0h] BYREF
  __m128i v61; // [rsp+30h] [rbp-390h] BYREF
  __int64 v62; // [rsp+40h] [rbp-380h] BYREF
  __m128i v63; // [rsp+48h] [rbp-378h] BYREF
  __int64 v64; // [rsp+58h] [rbp-368h]
  __m128i v65; // [rsp+60h] [rbp-360h] BYREF
  __m128i v66; // [rsp+70h] [rbp-350h]
  unsigned __int64 *v67; // [rsp+80h] [rbp-340h] BYREF
  __int64 v68; // [rsp+88h] [rbp-338h]
  _BYTE v69[320]; // [rsp+90h] [rbp-330h] BYREF
  char v70; // [rsp+1D0h] [rbp-1F0h]
  int v71; // [rsp+1D4h] [rbp-1ECh]
  __int64 v72; // [rsp+1D8h] [rbp-1E8h]
  void *v73; // [rsp+1E0h] [rbp-1E0h] BYREF
  __int32 v74; // [rsp+1E8h] [rbp-1D8h]
  __int8 v75; // [rsp+1ECh] [rbp-1D4h]
  __int64 v76; // [rsp+1F0h] [rbp-1D0h]
  __m128i v77; // [rsp+1F8h] [rbp-1C8h] BYREF
  __int64 v78; // [rsp+208h] [rbp-1B8h]
  __m128i v79; // [rsp+210h] [rbp-1B0h] BYREF
  __m128i v80; // [rsp+220h] [rbp-1A0h] BYREF
  unsigned __int64 *v81; // [rsp+230h] [rbp-190h] BYREF
  __int64 v82; // [rsp+238h] [rbp-188h]
  _BYTE v83[320]; // [rsp+240h] [rbp-180h] BYREF
  char v84; // [rsp+380h] [rbp-40h]
  int v85; // [rsp+384h] [rbp-3Ch]
  __int64 v86; // [rsp+388h] [rbp-38h]

  v4 = *a1;
  v5 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v5) )
  {
    v29 = sub_B2BE50(v4);
    v30 = sub_B6F970(v29);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v30 + 48LL))(v30) )
      return;
  }
  v9 = *(_QWORD *)(a2 + 104);
  v10 = *(__int64 **)(v9 + 32);
  v11 = *v10;
  if ( !*(_DWORD *)(a2 + 40) )
  {
    v58 = *v10;
    sub_D4BD20(&v59, v9, v6, v7, v8, v11);
    sub_B157E0((__int64)&v60, &v59);
    sub_B17640((__int64)&v73, (__int64)"loop-vectorize", (__int64)"MissedExplicitlyDisabled", 24, &v60, v58);
    sub_B18290((__int64)&v73, "loop not vectorized: vectorization is explicitly disabled", 0x39u);
    v67 = (unsigned __int64 *)v69;
    v25 = _mm_loadu_si128(&v77);
    v26 = _mm_load_si128(&v79);
    v61.m128i_i32[2] = v74;
    v27 = _mm_load_si128(&v80);
    v61.m128i_i64[0] = (__int64)&unk_49D9D40;
    v61.m128i_i8[12] = v75;
    v63 = v25;
    v62 = v76;
    v65 = v26;
    v64 = v78;
    v68 = 0x400000000LL;
    v66 = v27;
    if ( (_DWORD)v82 )
    {
      sub_31A82F0((__int64)&v67, (__int64)&v81, v22, (unsigned int)v82, v23, v24);
      v73 = &unk_49D9D40;
      v31 = v81;
      v70 = v84;
      v71 = v85;
      v72 = v86;
      v61.m128i_i64[0] = (__int64)&unk_49D9DB0;
      v32 = 10LL * (unsigned int)v82;
      v28 = &v81[v32];
      if ( v81 != &v81[v32] )
      {
        do
        {
          v28 -= 10;
          v33 = v28[4];
          if ( (unsigned __int64 *)v33 != v28 + 6 )
            j_j___libc_free_0(v33);
          if ( (unsigned __int64 *)*v28 != v28 + 2 )
            j_j___libc_free_0(*v28);
        }
        while ( v31 != v28 );
        v28 = v81;
      }
    }
    else
    {
      v28 = v81;
      v70 = v84;
      v71 = v85;
      v72 = v86;
      v61.m128i_i64[0] = (__int64)&unk_49D9DB0;
    }
    if ( v28 != (unsigned __int64 *)v83 )
      _libc_free((unsigned __int64)v28);
    if ( v59 )
      sub_B91220((__int64)&v59, v59);
    goto LABEL_9;
  }
  v57 = *v10;
  sub_D4BD20(&v60, v9, v6, v7, v8, v11);
  sub_B157E0((__int64)&v61, &v60);
  sub_B17640((__int64)&v73, (__int64)"loop-vectorize", (__int64)"MissedDetails", 13, &v61, v57);
  if ( v60.m128i_i64[0] )
    sub_B91220((__int64)&v60, v60.m128i_i64[0]);
  sub_B18290((__int64)&v73, "loop not vectorized", 0x13u);
  if ( *(_DWORD *)(a2 + 40) == 1 )
  {
    sub_B18290((__int64)&v73, " (Force=", 8u);
    v61.m128i_i64[0] = (__int64)&v62;
    sub_31A3B90(v61.m128i_i64, "Force", (__int64)"");
    v63.m128i_i64[1] = (__int64)&v65;
    sub_31A3B90(&v63.m128i_i64[1], "true", (__int64)"");
    v43 = &v61;
    v66 = 0u;
    sub_31A41F0((__int64)&v73, (__int64)&v61);
    if ( (__m128i *)v63.m128i_i64[1] != &v65 )
    {
      v43 = (__m128i *)(v65.m128i_i64[0] + 1);
      j_j___libc_free_0(v63.m128i_u64[1]);
    }
    if ( (__int64 *)v61.m128i_i64[0] != &v62 )
    {
      v43 = (__m128i *)(v62 + 1);
      j_j___libc_free_0(v61.m128i_u64[0]);
    }
    v47 = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)v47 )
    {
      sub_B18290((__int64)&v73, ", Vector Width=", 0xFu);
      v54 = *(_DWORD *)(a2 + 8);
      v60.m128i_i8[4] = *(_DWORD *)(a2 + 88) == 1;
      v60.m128i_i32[0] = v54;
      sub_B16C30((__int64)&v61, "VectorWidth", 11, v60.m128i_i64[0]);
      v43 = &v61;
      sub_31A41F0((__int64)&v73, (__int64)&v61);
      if ( (__m128i *)v63.m128i_i64[1] != &v65 )
      {
        v43 = (__m128i *)(v65.m128i_i64[0] + 1);
        j_j___libc_free_0(v63.m128i_u64[1]);
      }
      if ( (__int64 *)v61.m128i_i64[0] != &v62 )
      {
        v43 = (__m128i *)(v62 + 1);
        j_j___libc_free_0(v61.m128i_u64[0]);
      }
    }
    if ( *(_DWORD *)(a2 + 24) || (sub_F6E5D0(*(_QWORD *)(a2 + 104), (__int64)v43, v47, v44, v45, v46) & 2) != 0 )
    {
      sub_B18290((__int64)&v73, ", Interleave Count=", 0x13u);
      v51 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v51 )
        LODWORD(v51) = (sub_F6E5D0(*(_QWORD *)(a2 + 104), (__int64)", Interleave Count=", v48, v51, v49, v50) & 2) != 0;
      sub_B169E0(v61.m128i_i64, "InterleaveCount", 15, v51);
      sub_31A41F0((__int64)&v73, (__int64)&v61);
      if ( (__m128i *)v63.m128i_i64[1] != &v65 )
        j_j___libc_free_0(v63.m128i_u64[1]);
      if ( (__int64 *)v61.m128i_i64[0] != &v62 )
        j_j___libc_free_0(v61.m128i_u64[0]);
    }
    sub_B18290((__int64)&v73, ")", 1u);
  }
  v14 = _mm_loadu_si128(&v77);
  v15 = _mm_load_si128(&v79);
  v67 = (unsigned __int64 *)v69;
  v61.m128i_i32[2] = v74;
  v16 = _mm_load_si128(&v80);
  v17 = v82;
  v63 = v14;
  v61.m128i_i8[12] = v75;
  v18 = v81;
  v65 = v15;
  v62 = v76;
  v61.m128i_i64[0] = (__int64)&unk_49D9D40;
  v66 = v16;
  v64 = v78;
  v68 = 0x400000000LL;
  if ( !(_DWORD)v82 )
    goto LABEL_7;
  if ( v81 == (unsigned __int64 *)v83 )
  {
    v34 = (unsigned int)v82;
    v35 = (__m128i *)v69;
    if ( (unsigned int)v82 > 4 )
    {
      v55 = v82;
      sub_11F02D0((__int64)&v67, (unsigned int)v82, (__int64)v69, (unsigned int)v82, v12, v13);
      v35 = (__m128i *)v67;
      v18 = v81;
      v34 = (unsigned int)v82;
      v17 = v55;
    }
    v36 = &v18[10 * v34];
    if ( v36 == v18 )
    {
      LODWORD(v68) = v17;
    }
    else
    {
      v37 = (const __m128i *)(v18 + 6);
      v38 = v18 + 2;
      v39 = (unsigned __int64)v18 + (((char *)v36 - (char *)v18 - 80) & 0xFFFFFFFFFFFFFFF0LL) + 128;
      do
      {
        if ( v35 )
        {
          v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
          v42 = v37[-3].m128i_i64[0];
          if ( (unsigned __int64 *)v42 == v38 )
          {
            v35[1] = _mm_loadu_si128(v37 - 2);
          }
          else
          {
            v35->m128i_i64[0] = v42;
            v35[1].m128i_i64[0] = v37[-2].m128i_i64[0];
          }
          v35->m128i_i64[1] = v37[-3].m128i_i64[1];
          v37[-3].m128i_i64[0] = (__int64)v38;
          v37[-3].m128i_i64[1] = 0;
          v37[-2].m128i_i8[0] = 0;
          v35[2].m128i_i64[0] = (__int64)v35[3].m128i_i64;
          v40 = (const __m128i *)v37[-1].m128i_i64[0];
          if ( v37 == v40 )
          {
            v35[3] = _mm_loadu_si128(v37);
          }
          else
          {
            v35[2].m128i_i64[0] = (__int64)v40;
            v35[3].m128i_i64[0] = v37->m128i_i64[0];
          }
          v35[2].m128i_i64[1] = v37[-1].m128i_i64[1];
          v41 = _mm_loadu_si128(v37 + 1);
          v37[-1].m128i_i64[0] = (__int64)v37;
          v37[-1].m128i_i64[1] = 0;
          v37->m128i_i8[0] = 0;
          v35[4] = v41;
        }
        v37 += 5;
        v35 += 5;
        v38 += 10;
      }
      while ( (const __m128i *)v39 != v37 );
      LODWORD(v68) = v17;
      v56 = v81;
      v52 = 10LL * (unsigned int)v82;
      v18 = &v81[v52];
      if ( v81 != &v81[v52] )
      {
        do
        {
          v18 -= 10;
          v53 = v18[4];
          if ( (unsigned __int64 *)v53 != v18 + 6 )
            j_j___libc_free_0(v53);
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            j_j___libc_free_0(*v18);
        }
        while ( v56 != v18 );
        v18 = v81;
      }
    }
LABEL_7:
    v70 = v84;
    v71 = v85;
    v72 = v86;
    v61.m128i_i64[0] = (__int64)&unk_49D9DB0;
    if ( v18 != (unsigned __int64 *)v83 )
      _libc_free((unsigned __int64)v18);
    goto LABEL_9;
  }
  v67 = v81;
  v68 = v82;
  v70 = v84;
  v71 = v85;
  v72 = v86;
  v61.m128i_i64[0] = (__int64)&unk_49D9DB0;
LABEL_9:
  sub_1049740(a1, (__int64)&v61);
  v19 = v67;
  v61.m128i_i64[0] = (__int64)&unk_49D9D40;
  v20 = &v67[10 * (unsigned int)v68];
  if ( v67 != v20 )
  {
    do
    {
      v20 -= 10;
      v21 = v20[4];
      if ( (unsigned __int64 *)v21 != v20 + 6 )
        j_j___libc_free_0(v21);
      if ( (unsigned __int64 *)*v20 != v20 + 2 )
        j_j___libc_free_0(*v20);
    }
    while ( v19 != v20 );
    v20 = v67;
  }
  if ( v20 != (unsigned __int64 *)v69 )
    _libc_free((unsigned __int64)v20);
}
