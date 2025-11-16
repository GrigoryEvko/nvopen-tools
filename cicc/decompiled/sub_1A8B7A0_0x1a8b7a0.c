// Function: sub_1A8B7A0
// Address: 0x1a8b7a0
//
__int64 __fastcall sub_1A8B7A0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, size_t a5)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  _QWORD *v10; // r15
  _QWORD *v11; // r15
  _QWORD *v12; // r12
  _QWORD *v13; // rdi
  _QWORD *v14; // r15
  __int64 v15; // r12
  char *v16; // rsi
  _QWORD *v17; // r15
  _QWORD *v18; // r12
  _QWORD *v19; // rdi
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rax
  __m128i v25; // xmm2
  _QWORD *v26; // rbx
  _QWORD *v27; // r12
  _QWORD *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-498h]
  __int64 *v36; // [rsp+38h] [rbp-468h]
  char v37; // [rsp+57h] [rbp-449h]
  __int64 v38; // [rsp+68h] [rbp-438h] BYREF
  __m128i v39; // [rsp+70h] [rbp-430h] BYREF
  __int64 v40; // [rsp+80h] [rbp-420h]
  __m128i v41; // [rsp+90h] [rbp-410h] BYREF
  char v42; // [rsp+A0h] [rbp-400h]
  char v43; // [rsp+A1h] [rbp-3FFh]
  __m128i v44; // [rsp+B0h] [rbp-3F0h] BYREF
  __int64 v45; // [rsp+C0h] [rbp-3E0h] BYREF
  __m128i v46; // [rsp+C8h] [rbp-3D8h]
  __int64 v47; // [rsp+D8h] [rbp-3C8h]
  __int64 v48; // [rsp+E0h] [rbp-3C0h]
  __m128i v49; // [rsp+E8h] [rbp-3B8h]
  __int64 v50; // [rsp+F8h] [rbp-3A8h]
  char v51; // [rsp+100h] [rbp-3A0h]
  _BYTE *v52; // [rsp+108h] [rbp-398h] BYREF
  __int64 v53; // [rsp+110h] [rbp-390h]
  _BYTE v54[352]; // [rsp+118h] [rbp-388h] BYREF
  char v55; // [rsp+278h] [rbp-228h]
  int v56; // [rsp+27Ch] [rbp-224h]
  __int64 v57; // [rsp+280h] [rbp-220h]
  void *v58; // [rsp+290h] [rbp-210h] BYREF
  __int64 v59; // [rsp+298h] [rbp-208h]
  __int64 v60; // [rsp+2A0h] [rbp-200h]
  __m128i v61; // [rsp+2A8h] [rbp-1F8h] BYREF
  __int64 v62; // [rsp+2B8h] [rbp-1E8h]
  __int64 v63; // [rsp+2C0h] [rbp-1E0h]
  __m128i v64; // [rsp+2C8h] [rbp-1D8h] BYREF
  __int64 v65; // [rsp+2D8h] [rbp-1C8h]
  char v66; // [rsp+2E0h] [rbp-1C0h]
  _BYTE *v67; // [rsp+2E8h] [rbp-1B8h] BYREF
  __int64 v68; // [rsp+2F0h] [rbp-1B0h]
  _BYTE v69[352]; // [rsp+2F8h] [rbp-1A8h] BYREF
  char v70; // [rsp+458h] [rbp-48h]
  int v71; // [rsp+45Ch] [rbp-44h]
  __int64 v72; // [rsp+460h] [rbp-40h]

  v31 = sub_15E0530(a1[1]);
  v37 = *((_BYTE *)a1 + 57);
  if ( v37 )
    v37 = *((_BYTE *)a1 + 56);
  v36 = (__int64 *)a1[6];
  v6 = sub_15E0530(*v36);
  if ( sub_1602790(v6)
    || (v29 = sub_15E0530(*v36),
        v30 = sub_16033E0(v29),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v30 + 48LL))(v30)) )
  {
    v7 = **(_QWORD **)(*a1 + 32);
    sub_13FD840(&v39, *a1);
    sub_15C9090((__int64)&v41, &v39);
    sub_15CA540((__int64)&v58, (__int64)"loop-distribute", (__int64)"NotDistributed", 14, &v41, v7);
    sub_15CAB20((__int64)&v58, "loop not distributed: use -Rpass-analysis=loop-distribute for more info", 0x47u);
    v8 = _mm_loadu_si128(&v61);
    v9 = _mm_loadu_si128(&v64);
    v44.m128i_i32[2] = v59;
    v46 = v8;
    v44.m128i_i8[12] = BYTE4(v59);
    v49 = v9;
    v45 = v60;
    v47 = v62;
    v44.m128i_i64[0] = (__int64)&unk_49ECF68;
    v48 = v63;
    v51 = v66;
    if ( v66 )
      v50 = v65;
    v52 = v54;
    v53 = 0x400000000LL;
    if ( (_DWORD)v68 )
    {
      sub_1A8B510((__int64)&v52, (__int64)&v67);
      v21 = v67;
      v55 = v70;
      v56 = v71;
      v57 = v72;
      v44.m128i_i64[0] = (__int64)&unk_49ECFC8;
      v58 = &unk_49ECF68;
      v10 = &v67[88 * (unsigned int)v68];
      if ( v67 != (_BYTE *)v10 )
      {
        do
        {
          v10 -= 11;
          v22 = (_QWORD *)v10[4];
          if ( v22 != v10 + 6 )
            j_j___libc_free_0(v22, v10[6] + 1LL);
          if ( (_QWORD *)*v10 != v10 + 2 )
            j_j___libc_free_0(*v10, v10[2] + 1LL);
        }
        while ( v21 != v10 );
        v10 = v67;
      }
    }
    else
    {
      v10 = v67;
      v55 = v70;
      v56 = v71;
      v57 = v72;
      v44.m128i_i64[0] = (__int64)&unk_49ECFC8;
    }
    if ( v10 != (_QWORD *)v69 )
      _libc_free((unsigned __int64)v10);
    if ( v39.m128i_i64[0] )
      sub_161E7C0((__int64)&v39, v39.m128i_i64[0]);
    sub_143AA50(v36, (__int64)&v44);
    v11 = v52;
    v44.m128i_i64[0] = (__int64)&unk_49ECF68;
    v12 = &v52[88 * (unsigned int)v53];
    if ( v52 != (_BYTE *)v12 )
    {
      do
      {
        v12 -= 11;
        v13 = (_QWORD *)v12[4];
        if ( v13 != v12 + 6 )
          j_j___libc_free_0(v13, v12[6] + 1LL);
        if ( (_QWORD *)*v12 != v12 + 2 )
          j_j___libc_free_0(*v12, v12[2] + 1LL);
      }
      while ( v11 != v12 );
      v12 = v52;
    }
    if ( v12 != (_QWORD *)v54 )
      _libc_free((unsigned __int64)v12);
  }
  v14 = (_QWORD *)a1[6];
  v15 = **(_QWORD **)(*a1 + 32);
  sub_13FD840(&v41, *a1);
  sub_15C9090((__int64)&v44, &v41);
  v16 = "loop-distribute";
  if ( v37 )
    v16 = (char *)off_4C6F360;
  sub_15CA680((__int64)&v58, (__int64)v16, a2, a3, &v44, v15);
  sub_15CAB20((__int64)&v58, "loop not distributed: ", 0x16u);
  sub_15CAB20((__int64)&v58, a4, a5);
  sub_143AA50(v14, (__int64)&v58);
  v17 = v67;
  v58 = &unk_49ECF68;
  v18 = &v67[88 * (unsigned int)v68];
  if ( v67 != (_BYTE *)v18 )
  {
    do
    {
      v18 -= 11;
      v19 = (_QWORD *)v18[4];
      if ( v19 != v18 + 6 )
        j_j___libc_free_0(v19, v18[6] + 1LL);
      if ( (_QWORD *)*v18 != v18 + 2 )
        j_j___libc_free_0(*v18, v18[2] + 1LL);
    }
    while ( v17 != v18 );
    v18 = v67;
  }
  if ( v18 != (_QWORD *)v69 )
    _libc_free((unsigned __int64)v18);
  if ( v41.m128i_i64[0] )
    sub_161E7C0((__int64)&v41, v41.m128i_i64[0]);
  if ( v37 )
  {
    v23 = *a1;
    v43 = 1;
    v41.m128i_i64[0] = (__int64)"loop not distributed: failed explicitly specified loop distribution";
    v42 = 3;
    sub_13FD840(&v38, v23);
    sub_15C9090((__int64)&v39, &v38);
    v24 = a1[1];
    v70 = 0;
    v25 = _mm_loadu_si128(&v39);
    v63 = 0;
    v60 = v24;
    v59 = 0x10000000DLL;
    v62 = v40;
    v64.m128i_i64[0] = (__int64)byte_3F871B3;
    v61 = v25;
    v67 = v69;
    v68 = 0x400000000LL;
    v64.m128i_i64[1] = 0;
    v66 = 0;
    v58 = &unk_49F5BC8;
    v71 = -1;
    sub_16E2FC0(v44.m128i_i64, (__int64)&v41);
    sub_15CAB20((__int64)&v58, v44.m128i_i64[0], v44.m128i_u64[1]);
    if ( (__int64 *)v44.m128i_i64[0] != &v45 )
      j_j___libc_free_0(v44.m128i_i64[0], v45 + 1);
    v58 = &unk_49ED028;
    sub_16027F0(v31, (__int64)&v58);
    v26 = v67;
    v58 = &unk_49ECF68;
    v27 = &v67[88 * (unsigned int)v68];
    if ( v67 != (_BYTE *)v27 )
    {
      do
      {
        v27 -= 11;
        v28 = (_QWORD *)v27[4];
        if ( v28 != v27 + 6 )
          j_j___libc_free_0(v28, v27[6] + 1LL);
        if ( (_QWORD *)*v27 != v27 + 2 )
          j_j___libc_free_0(*v27, v27[2] + 1LL);
      }
      while ( v26 != v27 );
      v27 = v67;
    }
    if ( v27 != (_QWORD *)v69 )
      _libc_free((unsigned __int64)v27);
    if ( v38 )
      sub_161E7C0((__int64)&v38, v38);
  }
  return 0;
}
