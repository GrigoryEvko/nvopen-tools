// Function: sub_2CB1DA0
// Address: 0x2cb1da0
//
__int64 __fastcall sub_2CB1DA0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r10
  __int64 *v11; // r13
  __int64 v12; // rdx
  _QWORD *v13; // r15
  __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r10
  unsigned __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  unsigned __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // rdx
  unsigned __int64 v29; // r8
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // r15
  _QWORD *v39; // rax
  __int64 v40; // r13
  int v41; // ecx
  bool v42; // zf
  __m128i v43; // xmm0
  int v47; // [rsp+18h] [rbp-128h]
  __int64 v48; // [rsp+18h] [rbp-128h]
  __int64 v49; // [rsp+20h] [rbp-120h]
  __int64 v50; // [rsp+28h] [rbp-118h]
  __int64 *v51; // [rsp+28h] [rbp-118h]
  __int64 v52; // [rsp+28h] [rbp-118h]
  __int64 v53; // [rsp+28h] [rbp-118h]
  __int64 v54; // [rsp+28h] [rbp-118h]
  __m128i v55; // [rsp+40h] [rbp-100h] BYREF
  __int64 v56; // [rsp+50h] [rbp-F0h]
  _BYTE *v57; // [rsp+68h] [rbp-D8h] BYREF
  _QWORD *v58; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v59; // [rsp+78h] [rbp-C8h]
  _QWORD v60[2]; // [rsp+80h] [rbp-C0h] BYREF
  const char *v61; // [rsp+90h] [rbp-B0h] BYREF
  char v62; // [rsp+B0h] [rbp-90h]
  char v63; // [rsp+B1h] [rbp-8Fh]
  __int64 *v64; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-78h]
  _BYTE v66[112]; // [rsp+D0h] [rbp-70h] BYREF

  v7 = a2;
  v11 = *(__int64 **)(a5 + 40);
  v12 = *(_QWORD *)(a6 + 104);
  v13 = (_QWORD *)*v11;
  v58 = v60;
  v59 = 0x200000000LL;
  v14 = *(_QWORD *)(a7 + 16);
  v57 = (_BYTE *)v12;
  if ( v14 )
  {
    v60[0] = v12;
    v60[1] = v14;
    LODWORD(v59) = 2;
    v15 = sub_2CAF220(a3, (__int64 *)&v58, a4);
    v7 = a2;
    v57 = v15;
  }
  v64 = (__int64 *)v66;
  v65 = 0x800000000LL;
  v50 = v7;
  v16 = sub_2CB1590((_QWORD *)*v11, v7, (__int64 *)&v57, a6 + 8, (int *)a6, 0x10u);
  v18 = (unsigned int)v65;
  v19 = v50;
  v20 = (unsigned int)v65 + 1LL;
  if ( v20 > HIDWORD(v65) )
  {
    v48 = v50;
    v52 = v16;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v20, v17);
    v18 = (unsigned int)v65;
    v19 = v48;
    v16 = v52;
  }
  v64[v18] = v16;
  LODWORD(v65) = v65 + 1;
  v21 = sub_2CB1590((_QWORD *)*v11, v19, (__int64 *)&v57, a6 + 56, (int *)(a6 + 4), 8u);
  v23 = (unsigned int)v65;
  v24 = (unsigned int)v65 + 1LL;
  if ( v24 > HIDWORD(v65) )
  {
    v53 = v21;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v24, v22);
    v23 = (unsigned int)v65;
    v21 = v53;
  }
  v64[v23] = v21;
  LODWORD(v65) = v65 + 1;
  v25 = sub_BCCE00(v13, 1u);
  v26 = sub_ACD640(v25, 0, 0);
  v28 = (unsigned int)v65;
  v29 = (unsigned int)v65 + 1LL;
  if ( v29 > HIDWORD(v65) )
  {
    v54 = v26;
    sub_C8D5F0((__int64)&v64, v66, (unsigned int)v65 + 1LL, 8u, v29, v27);
    v28 = (unsigned int)v65;
    v26 = v54;
  }
  v64[v28] = v26;
  LODWORD(v65) = v65 + 1;
  v32 = sub_2CAF420(v13, a6, a7, (__int64 *)&v57);
  v33 = (unsigned int)v65;
  v34 = (unsigned int)v65 + 1LL;
  if ( v34 > HIDWORD(v65) )
  {
    sub_C8D5F0((__int64)&v64, v66, v34, 8u, v30, v31);
    v33 = (unsigned int)v65;
  }
  v64[v33] = v32;
  LODWORD(v65) = v65 + 1;
  if ( *(_DWORD *)a6 == 2 )
    v35 = (*(_DWORD *)(a6 + 4) == 2) + 8875;
  else
    v35 = (*(_DWORD *)(a6 + 4) == 2) + 8873;
  v36 = 0;
  v37 = sub_B6E160(v11, v35, 0, 0);
  v63 = 1;
  v38 = v37;
  v62 = 3;
  v61 = "idp2a";
  v51 = v64;
  v49 = (unsigned int)v65;
  if ( v37 )
    v36 = *(_QWORD *)(v37 + 24);
  v47 = v65 + 1;
  v39 = sub_BD2C40(88, (int)v65 + 1);
  v40 = (__int64)v39;
  if ( v39 )
  {
    sub_B44260((__int64)v39, **(_QWORD **)(v36 + 16), 56, v47 & 0x7FFFFFF, 0, 0);
    *(_QWORD *)(v40 + 72) = 0;
    sub_B4A290(v40, v36, v38, v51, v49, (__int64)&v61, 0, 0);
  }
  sub_B43DD0(v40, (__int64)v57);
  v41 = *(_DWORD *)(a6 + 112);
  *(_QWORD *)a1 = v40;
  v42 = *(_QWORD *)a6 == 0x200000002LL;
  *(_QWORD *)(a1 + 16) = v40;
  v57 = (_BYTE *)v40;
  *(_DWORD *)(a1 + 12) = v41;
  *(_DWORD *)(a1 + 8) = v42 + 1;
  if ( v41 != *(_DWORD *)(a7 + 12) )
  {
    sub_2CB0BD0((__int64)&v55, a3, a4, a1, a7);
    v43 = _mm_loadu_si128(&v55);
    *(_QWORD *)(a1 + 16) = v56;
    *(__m128i *)a1 = v43;
  }
  if ( v64 != (__int64 *)v66 )
    _libc_free((unsigned __int64)v64);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  return a1;
}
