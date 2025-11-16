// Function: sub_31A8E90
// Address: 0x31a8e90
//
void __fastcall sub_31A8E90(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  unsigned __int64 *v24; // r15
  unsigned __int64 *v25; // r13
  unsigned __int64 *v26; // r12
  unsigned __int64 v27; // rdi
  unsigned __int64 *v28; // r13
  __int64 v29; // r8
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-3A8h] BYREF
  __m128i v34; // [rsp+10h] [rbp-3A0h] BYREF
  void *v35; // [rsp+20h] [rbp-390h] BYREF
  int v36; // [rsp+28h] [rbp-388h]
  char v37; // [rsp+2Ch] [rbp-384h]
  __int64 v38; // [rsp+30h] [rbp-380h]
  __m128i v39; // [rsp+38h] [rbp-378h]
  __int64 v40; // [rsp+48h] [rbp-368h]
  __m128i v41; // [rsp+50h] [rbp-360h]
  __m128i v42; // [rsp+60h] [rbp-350h]
  unsigned __int64 *v43; // [rsp+70h] [rbp-340h] BYREF
  __int64 v44; // [rsp+78h] [rbp-338h]
  _BYTE v45[320]; // [rsp+80h] [rbp-330h] BYREF
  char v46; // [rsp+1C0h] [rbp-1F0h]
  int v47; // [rsp+1C4h] [rbp-1ECh]
  __int64 v48; // [rsp+1C8h] [rbp-1E8h]
  void *v49; // [rsp+1D0h] [rbp-1E0h] BYREF
  int v50; // [rsp+1D8h] [rbp-1D8h]
  char v51; // [rsp+1DCh] [rbp-1D4h]
  __int64 v52; // [rsp+1E0h] [rbp-1D0h]
  __m128i v53; // [rsp+1E8h] [rbp-1C8h] BYREF
  __int64 v54; // [rsp+1F8h] [rbp-1B8h]
  __m128i v55; // [rsp+200h] [rbp-1B0h] BYREF
  __m128i v56; // [rsp+210h] [rbp-1A0h] BYREF
  unsigned __int64 *v57; // [rsp+220h] [rbp-190h] BYREF
  unsigned int v58; // [rsp+228h] [rbp-188h]
  _BYTE v59[320]; // [rsp+230h] [rbp-180h] BYREF
  char v60; // [rsp+370h] [rbp-40h]
  int v61; // [rsp+374h] [rbp-3Ch]
  __int64 v62; // [rsp+378h] [rbp-38h]

  v5 = *a1;
  v6 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v6) )
  {
    v31 = sub_B2BE50(v5);
    v32 = sub_B6F970(v31);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v32 + 48LL))(v32) )
      return;
  }
  v11 = **(_QWORD **)(*a3 + 32);
  sub_D4BD20(&v33, *a3, v7, v8, v9, v10);
  sub_B157E0((__int64)&v34, &v33);
  v16 = sub_31A4B60(a2, (__int64)&v33, v12, v13, v14, v15);
  sub_B17850((__int64)&v49, (__int64)v16, (__int64)"AllDisabled", 11, &v34, v11);
  sub_B18290(
    (__int64)&v49,
    "loop not vectorized: vectorization and interleaving are explicitly disabled, or the loop has already been vectorized",
    0x74u);
  v21 = _mm_loadu_si128(&v53);
  v22 = _mm_loadu_si128(&v55);
  v23 = _mm_loadu_si128(&v56);
  v43 = (unsigned __int64 *)v45;
  v36 = v50;
  v39 = v21;
  v37 = v51;
  v41 = v22;
  v38 = v52;
  v42 = v23;
  v35 = &unk_49D9D40;
  v40 = v54;
  v44 = 0x400000000LL;
  if ( v58 )
  {
    sub_31A82F0((__int64)&v43, (__int64)&v57, v17, v18, v19, v20);
    v49 = &unk_49D9D40;
    v28 = v57;
    v46 = v60;
    v47 = v61;
    v48 = v62;
    v35 = &unk_49D9DE8;
    v29 = 10LL * v58;
    v24 = &v57[v29];
    if ( v57 != &v57[v29] )
    {
      do
      {
        v24 -= 10;
        v30 = v24[4];
        if ( (unsigned __int64 *)v30 != v24 + 6 )
          j_j___libc_free_0(v30);
        if ( (unsigned __int64 *)*v24 != v24 + 2 )
          j_j___libc_free_0(*v24);
      }
      while ( v28 != v24 );
      v24 = v57;
      if ( v57 == (unsigned __int64 *)v59 )
        goto LABEL_6;
      goto LABEL_5;
    }
  }
  else
  {
    v24 = v57;
    v46 = v60;
    v47 = v61;
    v48 = v62;
    v35 = &unk_49D9DE8;
  }
  if ( v24 != (unsigned __int64 *)v59 )
LABEL_5:
    _libc_free((unsigned __int64)v24);
LABEL_6:
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  sub_1049740(a1, (__int64)&v35);
  v25 = v43;
  v35 = &unk_49D9D40;
  v26 = &v43[10 * (unsigned int)v44];
  if ( v43 != v26 )
  {
    do
    {
      v26 -= 10;
      v27 = v26[4];
      if ( (unsigned __int64 *)v27 != v26 + 6 )
        j_j___libc_free_0(v27);
      if ( (unsigned __int64 *)*v26 != v26 + 2 )
        j_j___libc_free_0(*v26);
    }
    while ( v25 != v26 );
    v26 = v43;
  }
  if ( v26 != (unsigned __int64 *)v45 )
    _libc_free((unsigned __int64)v26);
}
