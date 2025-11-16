// Function: sub_2BE29D0
// Address: 0x2be29d0
//
__int64 __fastcall sub_2BE29D0(unsigned __int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, int a5)
{
  __int64 v9; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  void *v18; // rax
  size_t v19; // rdx
  void *v20; // rcx
  unsigned int v21; // r15d
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rdx
  const __m128i **v32; // rsi
  __int64 v33; // rax
  unsigned __int64 v34; // [rsp+8h] [rbp-148h]
  __int64 v35; // [rsp+8h] [rbp-148h]
  unsigned __int64 v36[3]; // [rsp+10h] [rbp-140h] BYREF
  unsigned __int64 v37; // [rsp+28h] [rbp-128h]
  unsigned __int64 v38; // [rsp+30h] [rbp-120h]
  __int64 v39; // [rsp+38h] [rbp-118h]
  __int64 v40; // [rsp+40h] [rbp-110h]
  _QWORD *v41; // [rsp+48h] [rbp-108h]
  unsigned __int64 *v42; // [rsp+50h] [rbp-100h]
  unsigned __int64 v43; // [rsp+58h] [rbp-F8h]
  __int64 v44; // [rsp+60h] [rbp-F0h]
  unsigned __int64 v45; // [rsp+68h] [rbp-E8h]
  __int64 v46; // [rsp+70h] [rbp-E0h]
  __int64 v47; // [rsp+78h] [rbp-D8h]
  int v48; // [rsp+80h] [rbp-D0h]
  unsigned __int8 v49; // [rsp+84h] [rbp-CCh]
  __m128i v50; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-B0h]
  unsigned __int64 v52; // [rsp+A8h] [rbp-A8h]
  unsigned __int64 v53; // [rsp+B0h] [rbp-A0h]
  __int64 v54; // [rsp+B8h] [rbp-98h]
  __int64 v55; // [rsp+C0h] [rbp-90h]
  _QWORD *v56; // [rsp+C8h] [rbp-88h]
  unsigned __int64 *v57; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v58; // [rsp+D8h] [rbp-78h]
  __int64 v59; // [rsp+E0h] [rbp-70h]
  __int64 v60; // [rsp+E8h] [rbp-68h]
  unsigned __int64 v61; // [rsp+F0h] [rbp-60h]
  __int64 v62; // [rsp+F8h] [rbp-58h]
  __int64 v63; // [rsp+100h] [rbp-50h]
  unsigned __int64 v64; // [rsp+108h] [rbp-48h]
  __int64 v65; // [rsp+110h] [rbp-40h]
  int v66; // [rsp+118h] [rbp-38h]

  if ( !*(_QWORD *)(a4 + 16) )
    return 0;
  a3[3] = a1;
  v9 = *(_QWORD *)(*(_QWORD *)(a4 + 16) + 40LL);
  v50 = 0u;
  LOBYTE(v51) = 0;
  sub_2BDEEE0(a3, (unsigned int)(v9 + 3), &v50);
  if ( (*(_DWORD *)a4 & 0x400) != 0 )
  {
    v11 = *(_QWORD **)(a4 + 16);
    v57 = a3;
    v50 = 0u;
    v51 = 0;
    v52 = 0;
    v53 = a1;
    v54 = a2;
    v55 = a4;
    v56 = v11;
    v12 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v11[8] - v11[7]) >> 4);
    if ( (__int64)(v11[8] - v11[7]) >= 0 )
    {
      v58 = 0;
      v13 = 0;
      v59 = 0;
      v14 = 16 * v12;
      v60 = 0;
      if ( v12 )
      {
        v15 = sub_22077B0(16 * v12);
        v13 = v15 + v14;
        v58 = v15;
        v60 = v15 + v14;
        do
        {
          if ( v15 )
          {
            *(_QWORD *)v15 = 0;
            *(_DWORD *)(v15 + 8) = 0;
          }
          v15 += 16;
        }
        while ( v15 != v13 );
        v11 = v56;
      }
      v59 = v13;
      v16 = v11[8] - v11[7];
      v17 = v11[4];
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v34 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 4);
      v18 = (void *)sub_2207820(v34);
      v19 = v34;
      v20 = v18;
      if ( v18 && (__int64)(v34 - 1) >= 0 )
      {
        if ( (__int64)(v34 - 2) < -1 )
          v19 = 1;
        v20 = memset(v18, 0, v19);
      }
      v64 = (unsigned __int64)v20;
      v65 = v17;
      if ( (a5 & 0x80u) != 0 )
        a5 &= 0xFFFFFFFA;
      v66 = a5;
      v52 = v53;
      v21 = sub_2BE23C0((__int64)&v50, 0);
      if ( v64 )
        j_j___libc_free_0_0(v64);
      v22 = v61;
      v35 = v62;
      if ( v62 != v61 )
      {
        do
        {
          v23 = *(_QWORD *)(v22 + 8);
          if ( v23 )
            j_j___libc_free_0(v23);
          v22 += 32LL;
        }
        while ( v35 != v22 );
        v22 = v61;
      }
      if ( v22 )
        j_j___libc_free_0(v22);
      if ( v58 )
        j_j___libc_free_0(v58);
      if ( v50.m128i_i64[0] )
        j_j___libc_free_0(v50.m128i_u64[0]);
      goto LABEL_30;
    }
LABEL_54:
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  }
  v27 = *(_QWORD **)(a4 + 16);
  v42 = a3;
  memset(v36, 0, sizeof(v36));
  v41 = v27;
  v28 = v27[8] - v27[7];
  v38 = a1;
  v39 = a2;
  v37 = 0;
  v40 = a4;
  if ( v28 < 0 )
    goto LABEL_54;
  v43 = 0;
  v44 = 0;
  v29 = 0xAAAAAAAAAAAAAAB0LL * (v28 >> 4);
  v45 = 0;
  if ( 0xAAAAAAAAAAAAAAABLL * (v28 >> 4) )
  {
    v30 = sub_22077B0(0xAAAAAAAAAAAAAAB0LL * (v28 >> 4));
    v31 = v30 + v29;
    v43 = v30;
    v45 = v30 + v29;
    do
    {
      if ( v30 )
      {
        *(_QWORD *)v30 = 0;
        *(_DWORD *)(v30 + 8) = 0;
      }
      v30 += 16;
    }
    while ( v31 != v30 );
    v32 = (const __m128i **)v42;
    v27 = v41;
  }
  else
  {
    v32 = (const __m128i **)a3;
    v31 = 0;
  }
  v44 = v31;
  v33 = v27[4];
  v49 = 0;
  v46 = v33;
  v47 = 0;
  if ( (a5 & 0x80u) != 0 )
    a5 &= 0xFFFFFFFA;
  v48 = a5;
  v37 = v38;
  sub_2BDBCA0(v36, v32, v31);
  sub_2BE13A0((__int64)v36, 0, v46);
  v21 = v49;
  if ( v43 )
    j_j___libc_free_0(v43);
  if ( v36[0] )
  {
    j_j___libc_free_0(v36[0]);
    if ( (_BYTE)v21 )
      goto LABEL_31;
LABEL_51:
    v51 = 0;
    v50.m128i_i64[1] = a2;
    v50.m128i_i64[0] = a2;
    sub_2BDEEE0(a3, 3u, &v50);
    return v21;
  }
LABEL_30:
  if ( !(_BYTE)v21 )
    goto LABEL_51;
LABEL_31:
  v24 = *a3;
  v25 = a3[1];
  if ( v25 != *a3 )
  {
    do
    {
      if ( !*(_BYTE *)(v24 + 16) )
      {
        *(_QWORD *)(v24 + 8) = a2;
        *(_QWORD *)v24 = a2;
      }
      v24 += 24LL;
    }
    while ( v25 != v24 );
    v24 = a3[1];
  }
  *(_BYTE *)(v24 - 32) = 0;
  *(_QWORD *)(v24 - 48) = a1;
  *(_QWORD *)(v24 - 40) = a1;
  *(_BYTE *)(v24 - 8) = 0;
  *(_QWORD *)(v24 - 24) = a2;
  *(_QWORD *)(v24 - 16) = a2;
  return v21;
}
