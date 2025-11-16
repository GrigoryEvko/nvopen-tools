// Function: sub_2A11DD0
// Address: 0x2a11dd0
//
void __fastcall sub_2A11DD0(__int64 *a1, __int64 *a2, unsigned int *a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // r15
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned int v19; // r13d
  __int64 v20; // rax
  int v21; // edx
  unsigned __int64 *v22; // r13
  __int64 v23; // r8
  unsigned __int64 *v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // r13
  unsigned __int64 *v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 *v32; // rcx
  __int64 v33; // r8
  __int64 v34; // [rsp+8h] [rbp-418h]
  __m128i *v35; // [rsp+10h] [rbp-410h]
  __int64 v36; // [rsp+18h] [rbp-408h]
  __int64 i; // [rsp+18h] [rbp-408h]
  __int64 v38; // [rsp+28h] [rbp-3F8h] BYREF
  __m128i v39; // [rsp+30h] [rbp-3F0h] BYREF
  __int64 v40[2]; // [rsp+40h] [rbp-3E0h] BYREF
  __int64 v41; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 *v42; // [rsp+60h] [rbp-3C0h]
  __int64 v43; // [rsp+70h] [rbp-3B0h] BYREF
  void *v44; // [rsp+90h] [rbp-390h] BYREF
  int v45; // [rsp+98h] [rbp-388h]
  char v46; // [rsp+9Ch] [rbp-384h]
  __int64 v47; // [rsp+A0h] [rbp-380h]
  __m128i v48; // [rsp+A8h] [rbp-378h]
  __int64 v49; // [rsp+B8h] [rbp-368h]
  __m128i v50; // [rsp+C0h] [rbp-360h]
  __m128i v51; // [rsp+D0h] [rbp-350h]
  __int64 *v52; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v53; // [rsp+E8h] [rbp-338h]
  _BYTE v54[324]; // [rsp+F0h] [rbp-330h] BYREF
  int v55; // [rsp+234h] [rbp-1ECh]
  __int64 v56; // [rsp+238h] [rbp-1E8h]
  _QWORD v57[10]; // [rsp+240h] [rbp-1E0h] BYREF
  unsigned __int64 *v58; // [rsp+290h] [rbp-190h]
  unsigned int v59; // [rsp+298h] [rbp-188h]
  char v60; // [rsp+2A0h] [rbp-180h] BYREF

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v6)
    || (v29 = sub_B2BE50(v5),
        v30 = sub_B6F970(v29),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v30 + 48LL))(v30)) )
  {
    v10 = *a2;
    v36 = **(_QWORD **)(v10 + 32);
    sub_D4BD20(&v38, v10, v7, v8, v9, v36);
    sub_B157E0((__int64)&v39, &v38);
    sub_B17430((__int64)v57, (__int64)"loop-unroll", (__int64)"FullyUnrolled", 13, &v39, v36);
    sub_B18290((__int64)v57, "completely unrolled loop with ", 0x1Eu);
    sub_B169E0(v40, "UnrollCount", 11, *a3);
    v11 = sub_23FD640((__int64)v57, (__int64)v40);
    sub_B18290(v11, " iterations", 0xBu);
    v14 = _mm_loadu_si128((const __m128i *)(v11 + 24));
    v15 = _mm_loadu_si128((const __m128i *)(v11 + 48));
    v45 = *(_DWORD *)(v11 + 8);
    v16 = _mm_loadu_si128((const __m128i *)(v11 + 64));
    v46 = *(_BYTE *)(v11 + 12);
    v17 = *(_QWORD *)(v11 + 16);
    v48 = v14;
    v47 = v17;
    v44 = &unk_49D9D40;
    v18 = *(_QWORD *)(v11 + 40);
    v53 = 0x400000000LL;
    v19 = *(_DWORD *)(v11 + 88);
    v49 = v18;
    v52 = (__int64 *)v54;
    v50 = v15;
    v51 = v16;
    if ( v19 && &v52 != (__int64 **)(v11 + 80) )
    {
      v31 = v19;
      v32 = (__int64 *)v54;
      if ( v19 > 4 )
      {
        sub_11F02D0((__int64)&v52, v19, v11 + 80, (__int64)v54, v12, v13);
        v32 = v52;
        v31 = *(unsigned int *)(v11 + 88);
      }
      v33 = *(_QWORD *)(v11 + 80);
      for ( i = v33 + 80 * v31; i != v33; v32 += 10 )
      {
        if ( v32 )
        {
          v34 = v33;
          *v32 = (__int64)(v32 + 2);
          v35 = (__m128i *)v32;
          sub_2A10890(v32, *(_BYTE **)v33, *(_QWORD *)v33 + *(_QWORD *)(v33 + 8));
          v35[2].m128i_i64[0] = (__int64)v35[3].m128i_i64;
          sub_2A10890(v35[2].m128i_i64, *(_BYTE **)(v34 + 32), *(_QWORD *)(v34 + 32) + *(_QWORD *)(v34 + 40));
          v33 = v34;
          v32 = (__int64 *)v35;
          v35[4] = _mm_loadu_si128((const __m128i *)(v34 + 64));
        }
        v33 += 80;
      }
      LODWORD(v53) = v19;
    }
    v20 = *(_QWORD *)(v11 + 424);
    v54[320] = *(_BYTE *)(v11 + 416);
    v21 = *(_DWORD *)(v11 + 420);
    v56 = v20;
    v55 = v21;
    v44 = &unk_49D9D78;
    if ( v42 != &v43 )
      j_j___libc_free_0((unsigned __int64)v42);
    if ( (__int64 *)v40[0] != &v41 )
      j_j___libc_free_0(v40[0]);
    v22 = v58;
    v57[0] = &unk_49D9D40;
    v23 = 10LL * v59;
    v24 = &v58[v23];
    if ( v58 != &v58[v23] )
    {
      do
      {
        v24 -= 10;
        v25 = v24[4];
        if ( (unsigned __int64 *)v25 != v24 + 6 )
          j_j___libc_free_0(v25);
        if ( (unsigned __int64 *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24);
      }
      while ( v22 != v24 );
      v24 = v58;
    }
    if ( v24 != (unsigned __int64 *)&v60 )
      _libc_free((unsigned __int64)v24);
    if ( v38 )
      sub_B91220((__int64)&v38, v38);
    sub_1049740(a1, (__int64)&v44);
    v26 = (unsigned __int64 *)v52;
    v44 = &unk_49D9D40;
    v27 = (unsigned __int64 *)&v52[10 * (unsigned int)v53];
    if ( v52 != (__int64 *)v27 )
    {
      do
      {
        v27 -= 10;
        v28 = v27[4];
        if ( (unsigned __int64 *)v28 != v27 + 6 )
          j_j___libc_free_0(v28);
        if ( (unsigned __int64 *)*v27 != v27 + 2 )
          j_j___libc_free_0(*v27);
      }
      while ( v26 != v27 );
      v27 = (unsigned __int64 *)v52;
    }
    if ( v27 != (unsigned __int64 *)v54 )
      _libc_free((unsigned __int64)v27);
  }
}
