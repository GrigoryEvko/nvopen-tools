// Function: sub_2B44ED0
// Address: 0x2b44ed0
//
void __fastcall sub_2B44ED0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 **a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r8
  int v14; // r9d
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r12
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 *v30; // r14
  unsigned __int64 *v31; // r12
  unsigned __int64 v32; // rdi
  unsigned __int64 *v33; // r13
  unsigned __int64 *v34; // r12
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // edx
  int v41; // r11d
  __int64 v42[2]; // [rsp+0h] [rbp-430h] BYREF
  __int64 v43; // [rsp+10h] [rbp-420h] BYREF
  __int64 *v44; // [rsp+20h] [rbp-410h]
  __int64 v45; // [rsp+30h] [rbp-400h] BYREF
  unsigned __int64 v46[2]; // [rsp+50h] [rbp-3E0h] BYREF
  __int64 v47; // [rsp+60h] [rbp-3D0h] BYREF
  __int64 *v48; // [rsp+70h] [rbp-3C0h]
  __int64 v49; // [rsp+80h] [rbp-3B0h] BYREF
  void *v50; // [rsp+A0h] [rbp-390h] BYREF
  int v51; // [rsp+A8h] [rbp-388h]
  char v52; // [rsp+ACh] [rbp-384h]
  __int64 v53; // [rsp+B0h] [rbp-380h]
  __m128i v54; // [rsp+B8h] [rbp-378h]
  __int64 v55; // [rsp+C8h] [rbp-368h]
  __m128i v56; // [rsp+D0h] [rbp-360h]
  __m128i v57; // [rsp+E0h] [rbp-350h]
  unsigned __int64 *v58; // [rsp+F0h] [rbp-340h] BYREF
  __int64 v59; // [rsp+F8h] [rbp-338h]
  _BYTE v60[324]; // [rsp+100h] [rbp-330h] BYREF
  int v61; // [rsp+244h] [rbp-1ECh]
  __int64 v62; // [rsp+248h] [rbp-1E8h]
  _QWORD v63[10]; // [rsp+250h] [rbp-1E0h] BYREF
  unsigned __int64 *v64; // [rsp+2A0h] [rbp-190h]
  unsigned int v65; // [rsp+2A8h] [rbp-188h]
  char v66; // [rsp+2B0h] [rbp-180h] BYREF

  v11 = *a1;
  v12 = sub_B2BE50(*a1);
  if ( !sub_B6EA50(v12) )
  {
    v37 = sub_B2BE50(v11);
    v38 = sub_B6F970(v37);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v38 + 48LL))(v38) )
      return;
  }
  if ( (*(_BYTE *)(a7 + 392) & 1) != 0 )
  {
    v13 = a7 + 400;
    v14 = 15;
  }
  else
  {
    v36 = *(unsigned int *)(a7 + 408);
    v13 = *(_QWORD *)(a7 + 400);
    if ( !(_DWORD)v36 )
      goto LABEL_39;
    v14 = v36 - 1;
  }
  v15 = **a8;
  v16 = v14 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v17 = (__int64 *)(v13 + 72LL * v16);
  v18 = *v17;
  if ( v15 != *v17 )
  {
    v40 = 1;
    while ( v18 != -4096 )
    {
      v41 = v40 + 1;
      v16 = v14 & (v40 + v16);
      v17 = (__int64 *)(v13 + 72LL * v16);
      v18 = *v17;
      if ( v15 == *v17 )
        goto LABEL_5;
      v40 = v41;
    }
    if ( (*(_BYTE *)(a7 + 392) & 1) != 0 )
    {
      v39 = 1152;
      goto LABEL_40;
    }
    v36 = *(unsigned int *)(a7 + 408);
LABEL_39:
    v39 = 72 * v36;
LABEL_40:
    v17 = (__int64 *)(v13 + v39);
  }
LABEL_5:
  sub_B174A0((__int64)v63, (__int64)"slp-vectorizer", (__int64)"VectorizedHorizontalReduction", 29, *(_QWORD *)v17[1]);
  sub_B18290((__int64)v63, "Vectorized horizontal reduction with cost ", 0x2Au);
  sub_B16D50((__int64)v46, "Cost", 4, *a9, a9[1]);
  v19 = sub_23FD640((__int64)v63, (__int64)v46);
  sub_B18290(v19, " and with tree size ", 0x14u);
  sub_B169E0(v42, "TreeSize", 8, *(_DWORD *)(a10 + 8));
  v24 = sub_23FD640(v19, (__int64)v42);
  v25 = _mm_loadu_si128((const __m128i *)(v24 + 24));
  v26 = _mm_loadu_si128((const __m128i *)(v24 + 48));
  v51 = *(_DWORD *)(v24 + 8);
  v27 = _mm_loadu_si128((const __m128i *)(v24 + 64));
  v52 = *(_BYTE *)(v24 + 12);
  v28 = *(_QWORD *)(v24 + 16);
  v54 = v25;
  v53 = v28;
  v50 = &unk_49D9D40;
  v29 = *(_QWORD *)(v24 + 40);
  v58 = (unsigned __int64 *)v60;
  v55 = v29;
  v59 = 0x400000000LL;
  LODWORD(v29) = *(_DWORD *)(v24 + 88);
  v56 = v26;
  v57 = v27;
  if ( (_DWORD)v29 )
    sub_2B44C50((__int64)&v58, v24 + 80, v20, v21, v22, v23);
  v60[320] = *(_BYTE *)(v24 + 416);
  v61 = *(_DWORD *)(v24 + 420);
  v62 = *(_QWORD *)(v24 + 424);
  v50 = &unk_49D9D78;
  if ( v44 != &v45 )
    j_j___libc_free_0((unsigned __int64)v44);
  if ( (__int64 *)v42[0] != &v43 )
    j_j___libc_free_0(v42[0]);
  if ( v48 != &v49 )
    j_j___libc_free_0((unsigned __int64)v48);
  if ( (__int64 *)v46[0] != &v47 )
    j_j___libc_free_0(v46[0]);
  v30 = v64;
  v63[0] = &unk_49D9D40;
  v31 = &v64[10 * v65];
  if ( v64 != v31 )
  {
    do
    {
      v31 -= 10;
      v32 = v31[4];
      if ( (unsigned __int64 *)v32 != v31 + 6 )
        j_j___libc_free_0(v32);
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
    }
    while ( v30 != v31 );
    v31 = v64;
  }
  if ( v31 != (unsigned __int64 *)&v66 )
    _libc_free((unsigned __int64)v31);
  sub_1049740(a1, (__int64)&v50);
  v33 = v58;
  v50 = &unk_49D9D40;
  v34 = &v58[10 * (unsigned int)v59];
  if ( v58 != v34 )
  {
    do
    {
      v34 -= 10;
      v35 = v34[4];
      if ( (unsigned __int64 *)v35 != v34 + 6 )
        j_j___libc_free_0(v35);
      if ( (unsigned __int64 *)*v34 != v34 + 2 )
        j_j___libc_free_0(*v34);
    }
    while ( v33 != v34 );
    v34 = v58;
  }
  if ( v34 != (unsigned __int64 *)v60 )
    _libc_free((unsigned __int64)v34);
}
