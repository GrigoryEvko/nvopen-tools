// Function: sub_34CF850
// Address: 0x34cf850
//
void __fastcall sub_34CF850(__int64 *a1, __int64 *a2, unsigned __int8 *a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  unsigned int v17; // ecx
  unsigned __int64 *v18; // rbx
  __int64 v19; // r8
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __m128i *v27; // r15
  __int64 v28; // rsi
  unsigned __int64 *v29; // rbx
  unsigned int v30; // [rsp+Ch] [rbp-414h]
  unsigned int v31; // [rsp+10h] [rbp-410h]
  unsigned __int64 *v32; // [rsp+10h] [rbp-410h]
  __int64 v33; // [rsp+18h] [rbp-408h]
  __int64 v34; // [rsp+28h] [rbp-3F8h] BYREF
  __m128i v35; // [rsp+30h] [rbp-3F0h] BYREF
  _BYTE *v36[2]; // [rsp+40h] [rbp-3E0h] BYREF
  __int64 v37; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 *v38; // [rsp+60h] [rbp-3C0h]
  __int64 v39; // [rsp+68h] [rbp-3B8h]
  __int64 v40; // [rsp+70h] [rbp-3B0h] BYREF
  __m128i v41; // [rsp+80h] [rbp-3A0h] BYREF
  __int64 *v42; // [rsp+90h] [rbp-390h] BYREF
  int v43; // [rsp+98h] [rbp-388h]
  char v44; // [rsp+9Ch] [rbp-384h]
  __int64 v45; // [rsp+A0h] [rbp-380h] BYREF
  __m128i v46; // [rsp+A8h] [rbp-378h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-368h]
  __m128i v48; // [rsp+C0h] [rbp-360h] BYREF
  __m128i v49; // [rsp+D0h] [rbp-350h]
  __m128i *v50; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v51; // [rsp+E8h] [rbp-338h]
  _BYTE v52[324]; // [rsp+F0h] [rbp-330h] BYREF
  int v53; // [rsp+234h] [rbp-1ECh]
  __int64 v54; // [rsp+238h] [rbp-1E8h]
  void *v55; // [rsp+240h] [rbp-1E0h] BYREF
  int v56; // [rsp+248h] [rbp-1D8h]
  char v57; // [rsp+24Ch] [rbp-1D4h]
  __int64 v58; // [rsp+250h] [rbp-1D0h]
  __m128i v59; // [rsp+258h] [rbp-1C8h] BYREF
  __int64 v60; // [rsp+268h] [rbp-1B8h]
  __m128i v61; // [rsp+270h] [rbp-1B0h] BYREF
  __m128i v62; // [rsp+280h] [rbp-1A0h] BYREF
  unsigned __int64 *v63; // [rsp+290h] [rbp-190h]
  unsigned int v64; // [rsp+298h] [rbp-188h]
  _BYTE v65[324]; // [rsp+2A0h] [rbp-180h] BYREF
  int v66; // [rsp+3E4h] [rbp-3Ch]
  __int64 v67; // [rsp+3E8h] [rbp-38h]

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v6)
    || (v25 = sub_B2BE50(v5),
        v26 = sub_B6F970(v25),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v26 + 48LL))(v26)) )
  {
    v10 = *a2;
    v33 = **(_QWORD **)(v10 + 32);
    sub_D4BD20(&v34, v10, v7, v8, v9, v33);
    sub_B157E0((__int64)&v35, &v34);
    sub_B17430((__int64)&v55, (__int64)"TTI", (__int64)"DontUnroll", 10, &v35, v33);
    sub_B18290((__int64)&v55, "advising against unrolling the loop because it contains a ", 0x3Au);
    sub_B16080((__int64)v36, "Call", 4, a3);
    v42 = &v45;
    sub_34CD8B0((__int64 *)&v42, v36[0], (__int64)&v36[0][(unsigned __int64)v36[1]]);
    v46.m128i_i64[1] = (__int64)&v48;
    sub_34CD8B0(&v46.m128i_i64[1], v38, (__int64)v38 + v39);
    v49 = _mm_loadu_si128(&v41);
    sub_B180C0((__int64)&v55, (unsigned __int64)&v42);
    if ( (__m128i *)v46.m128i_i64[1] != &v48 )
      j_j___libc_free_0(v46.m128i_u64[1]);
    if ( v42 != &v45 )
      j_j___libc_free_0((unsigned __int64)v42);
    v14 = _mm_loadu_si128(&v59);
    v15 = _mm_loadu_si128(&v61);
    v16 = _mm_loadu_si128(&v62);
    v43 = v56;
    v17 = v64;
    v46 = v14;
    v44 = v57;
    v48 = v15;
    v45 = v58;
    v49 = v16;
    v42 = (__int64 *)&unk_49D9D40;
    v47 = v60;
    v50 = (__m128i *)v52;
    v51 = 0x400000000LL;
    if ( v64 )
    {
      v27 = (__m128i *)v52;
      v28 = v64;
      if ( v64 > 4 )
      {
        v31 = v64;
        sub_11F02D0((__int64)&v50, v64, v11, v64, v12, v13);
        v27 = v50;
        v28 = v64;
        v17 = v31;
      }
      v29 = v63;
      v32 = &v63[10 * v28];
      if ( v63 != v32 )
      {
        do
        {
          if ( v27 )
          {
            v30 = v17;
            v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
            sub_34CD8B0(v27->m128i_i64, (_BYTE *)*v29, *v29 + v29[1]);
            v27[2].m128i_i64[0] = (__int64)v27[3].m128i_i64;
            sub_34CD8B0(v27[2].m128i_i64, (_BYTE *)v29[4], v29[4] + v29[5]);
            v17 = v30;
            v27[4] = _mm_loadu_si128((const __m128i *)v29 + 4);
          }
          v29 += 10;
          v27 += 5;
        }
        while ( v32 != v29 );
      }
      LODWORD(v51) = v17;
    }
    v52[320] = v65[320];
    v53 = v66;
    v54 = v67;
    v42 = (__int64 *)&unk_49D9D78;
    if ( v38 != &v40 )
      j_j___libc_free_0((unsigned __int64)v38);
    if ( (__int64 *)v36[0] != &v37 )
      j_j___libc_free_0((unsigned __int64)v36[0]);
    v18 = v63;
    v55 = &unk_49D9D40;
    v19 = 10LL * v64;
    v20 = &v63[v19];
    if ( v63 != &v63[v19] )
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
      while ( v18 != v20 );
      v20 = v63;
    }
    if ( v20 != (unsigned __int64 *)v65 )
      _libc_free((unsigned __int64)v20);
    if ( v34 )
      sub_B91220((__int64)&v34, v34);
    sub_1049740(a1, (__int64)&v42);
    v22 = (unsigned __int64 *)v50;
    v42 = (__int64 *)&unk_49D9D40;
    v23 = (unsigned __int64 *)&v50[5 * (unsigned int)v51];
    if ( v50 != (__m128i *)v23 )
    {
      do
      {
        v23 -= 10;
        v24 = v23[4];
        if ( (unsigned __int64 *)v24 != v23 + 6 )
          j_j___libc_free_0(v24);
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
          j_j___libc_free_0(*v23);
      }
      while ( v22 != v23 );
      v23 = (unsigned __int64 *)v50;
    }
    if ( v23 != (unsigned __int64 *)v52 )
      _libc_free((unsigned __int64)v23);
  }
}
