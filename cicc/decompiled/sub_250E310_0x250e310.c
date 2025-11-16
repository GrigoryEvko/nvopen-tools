// Function: sub_250E310
// Address: 0x250e310
//
void __fastcall sub_250E310(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  const char *v4; // rax
  char v5; // r14
  __int64 v6; // rax
  char v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  unsigned int *v17; // r14
  unsigned int *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // rbx
  __int64 v25; // r15
  unsigned __int8 *v26; // r14
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rdi
  const char *v30; // rax
  unsigned __int8 *v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rbx
  unsigned __int16 v34; // r14
  _QWORD *v35; // rax
  __int64 v36; // r15
  __int64 v37; // r10
  unsigned int v38; // r12d
  __int64 *v39; // rax
  unsigned int v40; // esi
  const char *v41; // r13
  unsigned __int16 v42; // bx
  _QWORD *v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // [rsp+0h] [rbp-120h]
  int v47; // [rsp+8h] [rbp-118h]
  __int64 v48; // [rsp+18h] [rbp-108h]
  __int64 v49; // [rsp+18h] [rbp-108h]
  __int64 v50; // [rsp+28h] [rbp-F8h]
  __int64 *v51; // [rsp+30h] [rbp-F0h]
  unsigned int v52; // [rsp+38h] [rbp-E8h]
  __int64 v53; // [rsp+38h] [rbp-E8h]
  __int64 v54; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int16 v55; // [rsp+48h] [rbp-D8h]
  unsigned int *v56; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+58h] [rbp-C8h]
  _BYTE v58[16]; // [rsp+60h] [rbp-C0h] BYREF
  const char *v59; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v60; // [rsp+78h] [rbp-A8h]
  __int16 v61; // [rsp+90h] [rbp-90h]
  char *v62; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v63; // [rsp+A8h] [rbp-78h]
  _BYTE v64[16]; // [rsp+B0h] [rbp-70h] BYREF
  __int16 v65; // [rsp+C0h] [rbp-60h]

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 24);
  v50 = *(_QWORD *)v2;
  v4 = sub_BD5D20(a1);
  v5 = *(_BYTE *)(a1 + 32);
  v62 = (char *)v4;
  v6 = *(_QWORD *)(a1 + 8);
  v65 = 261;
  v7 = v5 & 0xF;
  v63 = v8;
  v52 = *(_DWORD *)(v6 + 8) >> 8;
  v9 = sub_BD2DA0(136);
  v10 = v9;
  if ( v9 )
    sub_B2C3B0(v9, v3, v7, v52, (__int64)&v62, 0);
  v65 = 257;
  sub_BD6B50((unsigned __int8 *)a1, (const char **)&v62);
  sub_BA8540(v2 + 24, v10);
  v11 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(v10 + 64) = a1 + 56;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v10 + 56) = v11 | *(_QWORD *)(v10 + 56) & 7LL;
  *(_QWORD *)(v11 + 8) = v10 + 56;
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 56) & 7LL | (v10 + 56);
  *(_BYTE *)(v10 + 128) = *(_BYTE *)(v2 + 872);
  *(_WORD *)(a1 + 32) = *(_WORD *)(a1 + 32) & 0xBCC0 | 0x4007;
  sub_BD84D0(a1, v10);
  sub_B2F990(v10, *(_QWORD *)(a1 + 48), v12, v13);
  sub_B2F990(a1, 0, v14, v15);
  v16 = (__int64)&v56;
  v56 = (unsigned int *)v58;
  v57 = 0x100000000LL;
  sub_B9A9D0(a1, (__int64)&v56);
  v17 = v56;
  v18 = &v56[4 * (unsigned int)v57];
  if ( v56 != v18 )
  {
    do
    {
      v19 = *((_QWORD *)v17 + 1);
      v16 = *v17;
      v17 += 4;
      sub_B994D0(v10, v16, v19);
    }
    while ( v18 != v17 );
  }
  *(_QWORD *)(v10 + 120) = *(_QWORD *)(a1 + 120);
  v62 = "entry";
  v65 = 259;
  v53 = sub_22077B0(0x50u);
  if ( v53 )
  {
    v16 = v50;
    sub_AA4D50(v53, v50, (__int64)&v62, v10, 0);
  }
  v62 = v64;
  v63 = 0x800000000LL;
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    sub_B2C6D0(a1, v16, v20, v21);
  v22 = *(_QWORD *)(a1 + 96);
  if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
  {
    v49 = *(_QWORD *)(a1 + 96);
    sub_B2C6D0(v10, v16, v20, v21);
    v23 = *(_QWORD *)(v10 + 96);
    v22 = v49;
    v24 = v23 + 40LL * *(_QWORD *)(v10 + 104);
    if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v10, v16, v44, v45);
      v23 = *(_QWORD *)(v10 + 96);
      v22 = v49;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v10 + 96);
    v24 = v23 + 40LL * *(_QWORD *)(v10 + 104);
  }
  v25 = v22;
  v26 = (unsigned __int8 *)v23;
  if ( v23 != v24 )
  {
    do
    {
      v27 = (unsigned int)v63;
      v28 = (unsigned int)v63 + 1LL;
      if ( v28 > HIDWORD(v63) )
      {
        sub_C8D5F0((__int64)&v62, v64, v28, 8u, v23, v22);
        v27 = (unsigned int)v63;
      }
      v29 = v25;
      v25 += 40;
      *(_QWORD *)&v62[8 * v27] = v26;
      LODWORD(v63) = v63 + 1;
      v30 = sub_BD5D20(v29);
      v31 = v26;
      v26 += 40;
      v61 = 261;
      v59 = v30;
      v60 = v32;
      sub_BD6B50(v31, &v59);
    }
    while ( (unsigned __int8 *)v24 != v26 );
  }
  sub_B43C20((__int64)&v54, v53);
  v61 = 257;
  v33 = *(_QWORD *)(a1 + 24);
  v51 = (__int64 *)v62;
  v46 = v54;
  v34 = v55;
  v48 = (unsigned int)v63;
  v47 = v63 + 1;
  v35 = sub_BD2C40(88, (int)v63 + 1);
  v36 = (__int64)v35;
  if ( v35 )
  {
    sub_B44260((__int64)v35, **(_QWORD **)(v33 + 16), 56, v47 & 0x7FFFFFF, v46, v34);
    *(_QWORD *)(v36 + 72) = 0;
    sub_B4A290(v36, v33, a1, v51, v48, (__int64)&v59, 0, 0);
    v37 = v36;
  }
  else
  {
    v37 = 0;
  }
  v38 = 1;
  *(_WORD *)(v36 + 2) = *(_WORD *)(v36 + 2) & 0xFFFC | 1;
  v39 = (__int64 *)sub_BD5C60(v37);
  *(_QWORD *)(v36 + 72) = sub_A7A090((__int64 *)(v36 + 72), v39, -1, 31);
  sub_B43C20((__int64)&v59, v53);
  v40 = 1;
  v41 = v59;
  if ( *(_BYTE *)(*(_QWORD *)(v36 + 8) + 8LL) == 7 )
  {
    v38 = 0;
    v36 = 0;
    v40 = 0;
  }
  v42 = v60;
  v43 = sub_BD2C40(72, v40);
  if ( v43 )
    sub_B4BB80((__int64)v43, v50, v36, v38, (__int64)v41, v42);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( v56 != (unsigned int *)v58 )
    _libc_free((unsigned __int64)v56);
}
