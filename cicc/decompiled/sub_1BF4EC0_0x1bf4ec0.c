// Function: sub_1BF4EC0
// Address: 0x1bf4ec0
//
void __fastcall sub_1BF4EC0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 *v6; // rax
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  int v9; // ecx
  const __m128i *v10; // r15
  __m128i *v11; // rbx
  __m128i *v12; // r12
  __m128i *v13; // rdi
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  const __m128i *v16; // r15
  const __m128i *v17; // r13
  const __m128i *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  __m128i *v22; // rdx
  const __m128i *v23; // rdi
  const __m128i *v24; // rax
  const __m128i *v25; // r8
  const __m128i *v26; // rsi
  __m128i v27; // xmm4
  __int64 v28; // rsi
  const __m128i *v29; // rax
  const __m128i *v30; // rdi
  int v31; // [rsp+0h] [rbp-430h]
  const __m128i *v32; // [rsp+0h] [rbp-430h]
  const __m128i *v33; // [rsp+0h] [rbp-430h]
  __int64 v34; // [rsp+8h] [rbp-428h]
  __int64 v35; // [rsp+8h] [rbp-428h]
  __int64 v36; // [rsp+18h] [rbp-418h] BYREF
  __m128i v37[2]; // [rsp+20h] [rbp-410h] BYREF
  __m128i v38; // [rsp+40h] [rbp-3F0h] BYREF
  __int64 v39; // [rsp+50h] [rbp-3E0h] BYREF
  __m128i v40; // [rsp+58h] [rbp-3D8h] BYREF
  __int64 v41; // [rsp+68h] [rbp-3C8h]
  __int64 v42; // [rsp+70h] [rbp-3C0h] BYREF
  __m128i v43; // [rsp+78h] [rbp-3B8h]
  __int64 v44; // [rsp+88h] [rbp-3A8h]
  __int64 v45; // [rsp+90h] [rbp-3A0h]
  __m128i *v46; // [rsp+98h] [rbp-398h] BYREF
  __int64 v47; // [rsp+A0h] [rbp-390h]
  _BYTE v48[352]; // [rsp+A8h] [rbp-388h] BYREF
  char v49; // [rsp+208h] [rbp-228h]
  int v50; // [rsp+20Ch] [rbp-224h]
  __int64 v51; // [rsp+210h] [rbp-220h]
  void *v52; // [rsp+220h] [rbp-210h] BYREF
  __int32 v53; // [rsp+228h] [rbp-208h]
  __int8 v54; // [rsp+22Ch] [rbp-204h]
  __int64 v55; // [rsp+230h] [rbp-200h]
  __m128i v56; // [rsp+238h] [rbp-1F8h] BYREF
  __int64 v57; // [rsp+248h] [rbp-1E8h]
  __int64 v58; // [rsp+250h] [rbp-1E0h]
  __m128i v59; // [rsp+258h] [rbp-1D8h] BYREF
  __int64 v60; // [rsp+268h] [rbp-1C8h]
  char v61; // [rsp+270h] [rbp-1C0h]
  const __m128i *v62; // [rsp+278h] [rbp-1B8h] BYREF
  __int64 v63; // [rsp+280h] [rbp-1B0h]
  _BYTE v64[352]; // [rsp+288h] [rbp-1A8h] BYREF
  char v65; // [rsp+3E8h] [rbp-48h]
  int v66; // [rsp+3ECh] [rbp-44h]
  __int64 v67; // [rsp+3F0h] [rbp-40h]

  v4 = sub_15E0530(*a1);
  if ( !sub_1602790(v4) )
  {
    v19 = sub_15E0530(*a1);
    v20 = sub_16033E0(v19);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 48LL))(v20) )
      return;
  }
  v5 = *(_QWORD *)(a2 + 72);
  v6 = *(__int64 **)(v5 + 32);
  if ( !*(_DWORD *)(a2 + 40) )
  {
    v35 = *v6;
    sub_13FD840(&v36, v5);
    sub_15C9090((__int64)v37, &v36);
    sub_15CA540((__int64)&v52, (__int64)"loop-vectorize", (__int64)"MissedExplicitlyDisabled", 24, v37, v35);
    sub_15CAB20((__int64)&v52, "loop not vectorized: vectorization is explicitly disabled", 0x39u);
    v14 = _mm_loadu_si128(&v56);
    v15 = _mm_loadu_si128(&v59);
    v38.m128i_i32[2] = v53;
    v40 = v14;
    v38.m128i_i8[12] = v54;
    v43 = v15;
    v39 = v55;
    v41 = v57;
    v38.m128i_i64[0] = (__int64)&unk_49ECF68;
    v42 = v58;
    LOBYTE(v45) = v61;
    if ( v61 )
      v44 = v60;
    v46 = (__m128i *)v48;
    v47 = 0x400000000LL;
    if ( (_DWORD)v63 )
    {
      sub_1BF40D0((__int64)&v46, (__int64)&v62);
      v17 = v62;
      v49 = v65;
      v50 = v66;
      v51 = v67;
      v38.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v52 = &unk_49ECF68;
      v16 = (const __m128i *)((char *)v62 + 88 * (unsigned int)v63);
      if ( v62 != v16 )
      {
        do
        {
          v16 = (const __m128i *)((char *)v16 - 88);
          v18 = (const __m128i *)v16[2].m128i_i64[0];
          if ( v18 != &v16[3] )
            j_j___libc_free_0(v18, v16[3].m128i_i64[0] + 1);
          if ( (const __m128i *)v16->m128i_i64[0] != &v16[1] )
            j_j___libc_free_0(v16->m128i_i64[0], v16[1].m128i_i64[0] + 1);
        }
        while ( v17 != v16 );
        v16 = v62;
      }
    }
    else
    {
      v16 = v62;
      v49 = v65;
      v50 = v66;
      v51 = v67;
      v38.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    if ( v16 != (const __m128i *)v64 )
      _libc_free((unsigned __int64)v16);
    if ( v36 )
      sub_161E7C0((__int64)&v36, v36);
    goto LABEL_11;
  }
  v34 = *v6;
  sub_13FD840(v37, v5);
  sub_15C9090((__int64)&v38, v37);
  sub_15CA540((__int64)&v52, (__int64)"loop-vectorize", (__int64)"MissedDetails", 13, &v38, v34);
  if ( v37[0].m128i_i64[0] )
    sub_161E7C0((__int64)v37, v37[0].m128i_i64[0]);
  sub_15CAB20((__int64)&v52, "loop not vectorized", 0x13u);
  if ( *(_DWORD *)(a2 + 40) == 1 )
  {
    sub_15CAB20((__int64)&v52, " (Force=", 8u);
    v38.m128i_i64[0] = (__int64)&v39;
    sub_1BF09F0(v38.m128i_i64, "Force", (__int64)"");
    v40.m128i_i64[1] = (__int64)&v42;
    sub_1BF09F0(&v40.m128i_i64[1], "true", (__int64)"");
    v43.m128i_i64[1] = 0;
    v44 = 0;
    v45 = 0;
    sub_1BF0EE0((__int64)&v52, (__int64)&v38);
    if ( (__int64 *)v40.m128i_i64[1] != &v42 )
      j_j___libc_free_0(v40.m128i_i64[1], v42 + 1);
    if ( (__int64 *)v38.m128i_i64[0] != &v39 )
      j_j___libc_free_0(v38.m128i_i64[0], v39 + 1);
    if ( *(_DWORD *)(a2 + 8) )
    {
      sub_15CAB20((__int64)&v52, ", Vector Width=", 0xFu);
      sub_15C9C50((__int64)&v38, "VectorWidth", 11, *(_DWORD *)(a2 + 8));
      sub_1BF0EE0((__int64)&v52, (__int64)&v38);
      if ( (__int64 *)v40.m128i_i64[1] != &v42 )
        j_j___libc_free_0(v40.m128i_i64[1], v42 + 1);
      if ( (__int64 *)v38.m128i_i64[0] != &v39 )
        j_j___libc_free_0(v38.m128i_i64[0], v39 + 1);
    }
    if ( *(_DWORD *)(a2 + 24) )
    {
      sub_15CAB20((__int64)&v52, ", Interleave Count=", 0x13u);
      sub_15C9C50((__int64)&v38, "InterleaveCount", 15, *(_DWORD *)(a2 + 24));
      sub_1BF0EE0((__int64)&v52, (__int64)&v38);
      if ( (__int64 *)v40.m128i_i64[1] != &v42 )
        j_j___libc_free_0(v40.m128i_i64[1], v42 + 1);
      if ( (__int64 *)v38.m128i_i64[0] != &v39 )
        j_j___libc_free_0(v38.m128i_i64[0], v39 + 1);
    }
    sub_15CAB20((__int64)&v52, ")", 1u);
  }
  v7 = _mm_loadu_si128(&v56);
  v8 = _mm_loadu_si128(&v59);
  v38.m128i_i32[2] = v53;
  v40 = v7;
  v38.m128i_i8[12] = v54;
  v43 = v8;
  v39 = v55;
  v41 = v57;
  v38.m128i_i64[0] = (__int64)&unk_49ECF68;
  v42 = v58;
  LOBYTE(v45) = v61;
  if ( v61 )
    v44 = v60;
  v9 = v63;
  v10 = v62;
  v47 = 0x400000000LL;
  v46 = (__m128i *)v48;
  if ( !(_DWORD)v63 )
    goto LABEL_9;
  if ( v62 == (const __m128i *)v64 )
  {
    v21 = (unsigned int)v63;
    v22 = (__m128i *)v48;
    if ( (unsigned int)v63 > 4 )
    {
      v31 = v63;
      sub_14B3F20((__int64)&v46, (unsigned int)v63);
      v22 = v46;
      v10 = v62;
      v21 = (unsigned int)v63;
      v9 = v31;
    }
    v23 = (const __m128i *)((char *)v10 + 88 * v21);
    if ( v23 == v10 )
    {
      LODWORD(v47) = v9;
    }
    else
    {
      v24 = v10 + 3;
      v25 = v10 + 1;
      while ( 1 )
      {
        if ( v22 )
        {
          v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
          v28 = v24[-3].m128i_i64[0];
          if ( (const __m128i *)v28 == v25 )
          {
            v22[1] = _mm_loadu_si128(v24 - 2);
          }
          else
          {
            v22->m128i_i64[0] = v28;
            v22[1].m128i_i64[0] = v24[-2].m128i_i64[0];
          }
          v22->m128i_i64[1] = v24[-3].m128i_i64[1];
          v24[-3].m128i_i64[0] = (__int64)v25;
          v24[-3].m128i_i64[1] = 0;
          v24[-2].m128i_i8[0] = 0;
          v22[2].m128i_i64[0] = (__int64)v22[3].m128i_i64;
          v26 = (const __m128i *)v24[-1].m128i_i64[0];
          if ( v24 == v26 )
          {
            v22[3] = _mm_loadu_si128(v24);
          }
          else
          {
            v22[2].m128i_i64[0] = (__int64)v26;
            v22[3].m128i_i64[0] = v24->m128i_i64[0];
          }
          v22[2].m128i_i64[1] = v24[-1].m128i_i64[1];
          v27 = _mm_loadu_si128(v24 + 1);
          v24[-1].m128i_i64[0] = (__int64)v24;
          v24[-1].m128i_i64[1] = 0;
          v24->m128i_i8[0] = 0;
          v22[4] = v27;
          v22[5].m128i_i64[0] = v24[2].m128i_i64[0];
        }
        v22 = (__m128i *)((char *)v22 + 88);
        v25 = (const __m128i *)((char *)v25 + 88);
        if ( v23 == (const __m128i *)&v24[2].m128i_u64[1] )
          break;
        v24 = (const __m128i *)((char *)v24 + 88);
      }
      v10 = v62;
      LODWORD(v47) = v9;
      v29 = (const __m128i *)((char *)v62 + 88 * (unsigned int)v63);
      if ( v62 != v29 )
      {
        do
        {
          v29 = (const __m128i *)((char *)v29 - 88);
          v30 = (const __m128i *)v29[2].m128i_i64[0];
          if ( v30 != &v29[3] )
          {
            v32 = v29;
            j_j___libc_free_0(v30, v29[3].m128i_i64[0] + 1);
            v29 = v32;
          }
          if ( (const __m128i *)v29->m128i_i64[0] != &v29[1] )
          {
            v33 = v29;
            j_j___libc_free_0(v29->m128i_i64[0], v29[1].m128i_i64[0] + 1);
            v29 = v33;
          }
        }
        while ( v10 != v29 );
        v10 = v62;
      }
    }
LABEL_9:
    v49 = v65;
    v50 = v66;
    v51 = v67;
    v38.m128i_i64[0] = (__int64)&unk_49ECFC8;
    if ( v10 != (const __m128i *)v64 )
      _libc_free((unsigned __int64)v10);
    goto LABEL_11;
  }
  v46 = (__m128i *)v62;
  v47 = v63;
  v49 = v65;
  v50 = v66;
  v51 = v67;
  v38.m128i_i64[0] = (__int64)&unk_49ECFC8;
LABEL_11:
  sub_143AA50(a1, (__int64)&v38);
  v11 = v46;
  v38.m128i_i64[0] = (__int64)&unk_49ECF68;
  v12 = (__m128i *)((char *)v46 + 88 * (unsigned int)v47);
  if ( v46 != v12 )
  {
    do
    {
      v12 = (__m128i *)((char *)v12 - 88);
      v13 = (__m128i *)v12[2].m128i_i64[0];
      if ( v13 != &v12[3] )
        j_j___libc_free_0(v13, v12[3].m128i_i64[0] + 1);
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        j_j___libc_free_0(v12->m128i_i64[0], v12[1].m128i_i64[0] + 1);
    }
    while ( v11 != v12 );
    v12 = v46;
  }
  if ( v12 != (__m128i *)v48 )
    _libc_free((unsigned __int64)v12);
}
