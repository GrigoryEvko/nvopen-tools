// Function: sub_187E2A0
// Address: 0x187e2a0
//
unsigned __int64 __fastcall sub_187E2A0(
        __int64 *a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // rax
  int v18; // esi
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // edi
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // r15
  __int64 v33; // rdi
  size_t v34; // r12
  char *v35; // r14
  __int64 v36; // rax
  __int64 v37; // rcx
  unsigned __int8 *v38; // r8
  int v39; // r14d
  _QWORD *v40; // r15
  __int64 **v41; // r13
  __int64 v42; // r12
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  double v45; // xmm4_8
  double v46; // xmm5_8
  _BYTE *v47; // rax
  char *v48; // r13
  size_t v49; // r12
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rax
  __int64 v53; // r12
  unsigned __int64 result; // rax
  unsigned __int64 v55; // r15
  __int64 v56; // rdx
  __int64 *v57; // r15
  int v58; // ebx
  __int64 v59; // r14
  __int64 v60; // r12
  __int64 *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // rdi
  __int64 v68; // rax
  double v69; // xmm4_8
  double v70; // xmm5_8
  __int64 v71; // rdx
  __int64 *v72; // [rsp+8h] [rbp-E8h]
  size_t v73; // [rsp+8h] [rbp-E8h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+10h] [rbp-E0h]
  char *v76; // [rsp+18h] [rbp-D8h]
  __int64 v77; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v78; // [rsp+20h] [rbp-D0h]
  char *v79; // [rsp+20h] [rbp-D0h]
  __int64 v80; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v81; // [rsp+28h] [rbp-C8h]
  __int64 *v82; // [rsp+28h] [rbp-C8h]
  __int64 *v83[2]; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int8 v84[16]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v85; // [rsp+50h] [rbp-A0h]
  void *src; // [rsp+60h] [rbp-90h] BYREF
  __int64 v87; // [rsp+68h] [rbp-88h]
  __int64 v88; // [rsp+70h] [rbp-80h]
  __int128 v89; // [rsp+78h] [rbp-78h]
  __int128 v90; // [rsp+88h] [rbp-68h]
  __int128 v91; // [rsp+98h] [rbp-58h]
  __int128 v92; // [rsp+A8h] [rbp-48h]

  v9 = a1;
  v10 = a1[18];
  v76 = (char *)a1[19];
  v11 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v76[-v10] >> 4);
  if ( (__int64)&v76[-v10] <= 0 )
  {
LABEL_46:
    sub_1878A30(v10, v76);
    j_j___libc_free_0(0, 0);
  }
  else
  {
    while ( 1 )
    {
      v12 = 80 * v11;
      v80 = 80 * v11;
      v13 = sub_2207800(80 * v11, &unk_435FF63);
      v14 = v13;
      if ( v13 )
        break;
      v11 >>= 1;
      if ( !v11 )
        goto LABEL_46;
    }
    v74 = v11;
    v15 = v13 + 8;
    v16 = v13 + v12;
    v77 = v10 + 8;
    v17 = *(_QWORD *)(v10 + 16);
    if ( v17 )
    {
      v18 = *(_DWORD *)(v10 + 8);
      *(_QWORD *)(v14 + 16) = v17;
      *(_DWORD *)(v14 + 8) = v18;
      *(_QWORD *)(v14 + 24) = *(_QWORD *)(v10 + 24);
      *(_QWORD *)(v14 + 32) = *(_QWORD *)(v10 + 32);
      *(_QWORD *)(v17 + 8) = v15;
      v19 = *(_QWORD *)(v10 + 40);
      *(_QWORD *)(v10 + 16) = 0;
      *(_QWORD *)(v14 + 40) = v19;
      *(_QWORD *)(v10 + 40) = 0;
      *(_QWORD *)(v10 + 24) = v77;
      *(_QWORD *)(v10 + 32) = v77;
    }
    else
    {
      *(_DWORD *)(v14 + 8) = 0;
      *(_QWORD *)(v14 + 16) = 0;
      *(_QWORD *)(v14 + 24) = v15;
      *(_QWORD *)(v14 + 32) = v15;
      *(_QWORD *)(v14 + 40) = 0;
    }
    *(_QWORD *)(v14 + 48) = *(_QWORD *)(v10 + 48);
    *(_QWORD *)(v14 + 56) = *(_QWORD *)(v10 + 56);
    *(_QWORD *)(v14 + 64) = *(_QWORD *)(v10 + 64);
    *(_QWORD *)(v14 + 72) = *(_QWORD *)(v10 + 72);
    v20 = v14 + 80;
    if ( v16 == v14 + 80 )
    {
      v71 = v14;
    }
    else
    {
      do
      {
        v24 = *(_QWORD *)(v20 - 64);
        v25 = v20 - 72;
        v26 = v20 + 8;
        if ( v24 )
        {
          v21 = *(_DWORD *)(v20 - 72);
          *(_QWORD *)(v20 + 16) = v24;
          *(_DWORD *)(v20 + 8) = v21;
          *(_QWORD *)(v20 + 24) = *(_QWORD *)(v20 - 56);
          *(_QWORD *)(v20 + 32) = *(_QWORD *)(v20 - 48);
          *(_QWORD *)(v24 + 8) = v26;
          v22 = *(_QWORD *)(v20 - 40);
          *(_QWORD *)(v20 - 64) = 0;
          *(_QWORD *)(v20 + 40) = v22;
          *(_QWORD *)(v20 - 56) = v25;
          *(_QWORD *)(v20 - 48) = v25;
          *(_QWORD *)(v20 - 40) = 0;
        }
        else
        {
          *(_DWORD *)(v20 + 8) = 0;
          *(_QWORD *)(v20 + 16) = 0;
          *(_QWORD *)(v20 + 24) = v26;
          *(_QWORD *)(v20 + 32) = v26;
          *(_QWORD *)(v20 + 40) = 0;
        }
        v23 = *(_QWORD *)(v20 - 32);
        v20 += 80;
        *(_QWORD *)(v20 - 32) = v23;
        *(_QWORD *)(v20 - 24) = *(_QWORD *)(v20 - 104);
        *(_QWORD *)(v20 - 16) = *(_QWORD *)(v20 - 96);
        *(_QWORD *)(v20 - 8) = *(_QWORD *)(v20 - 88);
      }
      while ( v16 != v20 );
      v71 = v14 + 16 * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)(v12 - 160) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 5);
    }
    if ( *(_QWORD *)(v10 + 16) )
    {
      v72 = v9;
      v27 = *(_QWORD *)(v10 + 16);
      v28 = v71;
      do
      {
        sub_1876060(*(_QWORD *)(v27 + 24));
        v29 = v27;
        v27 = *(_QWORD *)(v27 + 16);
        j_j___libc_free_0(v29, 40);
      }
      while ( v27 );
      v9 = v72;
      v71 = v28;
    }
    *(_QWORD *)(v10 + 16) = 0;
    *(_QWORD *)(v10 + 40) = 0;
    *(_QWORD *)(v10 + 24) = v77;
    *(_QWORD *)(v10 + 32) = v77;
    if ( *(_QWORD *)(v71 + 16) )
    {
      *(_DWORD *)(v10 + 8) = *(_DWORD *)(v71 + 8);
      v30 = *(_QWORD *)(v71 + 16);
      *(_QWORD *)(v10 + 16) = v30;
      *(_QWORD *)(v10 + 24) = *(_QWORD *)(v71 + 24);
      *(_QWORD *)(v10 + 32) = *(_QWORD *)(v71 + 32);
      *(_QWORD *)(v30 + 8) = v77;
      *(_QWORD *)(v10 + 40) = *(_QWORD *)(v71 + 40);
      *(_QWORD *)(v71 + 16) = 0;
      *(_QWORD *)(v71 + 24) = v71 + 8;
      *(_QWORD *)(v71 + 32) = v71 + 8;
      *(_QWORD *)(v71 + 40) = 0;
    }
    v31 = v14;
    *(_QWORD *)(v10 + 48) = *(_QWORD *)(v71 + 48);
    *(_QWORD *)(v10 + 56) = *(_QWORD *)(v71 + 56);
    *(_QWORD *)(v10 + 64) = *(_QWORD *)(v71 + 64);
    *(_QWORD *)(v10 + 72) = *(_QWORD *)(v71 + 72);
    sub_1879B60(v10, v76, v14, v74);
    do
    {
      v32 = *(_QWORD *)(v31 + 16);
      while ( v32 )
      {
        sub_1876060(*(_QWORD *)(v32 + 24));
        v33 = v32;
        v32 = *(_QWORD *)(v32 + 16);
        j_j___libc_free_0(v33, 40);
      }
      v31 += 80;
    }
    while ( v16 != v31 );
    j_j___libc_free_0(v14, v80);
  }
  if ( v9[19] - v9[18] < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( 0xCCCCCCCCCCCCCCCDLL * ((v9[19] - v9[18]) >> 4) )
  {
    v34 = 0x6666666666666668LL * ((v9[19] - v9[18]) >> 4);
    v35 = (char *)sub_22077B0(v34);
    if ( v35 == &v35[v34] )
    {
      v73 = 0;
    }
    else
    {
      memset(v35, 0, v34);
      v73 = v34;
    }
  }
  else
  {
    v73 = 0;
    v35 = 0;
  }
  v88 = 0;
  v36 = v9[18];
  src = 0;
  v87 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  if ( v9[19] == v36 )
  {
    v49 = 0;
    v48 = 0;
  }
  else
  {
    v37 = 0;
    v38 = v84;
    v78 = (unsigned __int64 *)v35;
    v39 = 0;
    do
    {
      v81 = v38;
      v40 = (_QWORD *)(v36 + 80 * v37);
      sub_187E1D0((__int64 *)&src, (__int64)v40, v40[6], &v78[v37], v38);
      v41 = (__int64 **)v9[7];
      v42 = v40[8];
      v43 = sub_159C470(v9[6], v84[0], 0);
      v44 = sub_15A3BA0(v43, v41, 0);
      sub_164D160(v42, v44, (__m128)0LL, a3, a4, a5, v45, v46, a8, a9);
      sub_15E55B0(v40[8]);
      v47 = (_BYTE *)v40[9];
      v38 = v81;
      if ( v47 )
        *v47 = v84[0];
      v36 = v9[18];
      v37 = (unsigned int)++v39;
    }
    while ( v39 != 0xCCCCCCCCCCCCCCCDLL * ((v9[19] - v36) >> 4) );
    v48 = (char *)src;
    v35 = (char *)v78;
    v49 = v87 - (_QWORD)src;
  }
  v50 = (__int64 *)sub_1644C60(*(_QWORD **)*v9, 8u);
  v51 = sub_1645D80(v50, v49);
  v52 = (__int64 *)sub_15991C0(v48, v49, (__int64 **)v51);
  v53 = *v52;
  v85 = 257;
  v82 = v52;
  result = (unsigned __int64)sub_1648A60(88, 1u);
  v55 = result;
  if ( result )
    result = sub_15E51E0(result, *v9, v53, 1, 8, (__int64)v82, (__int64)v84, 0, 0, 0, 0);
  v56 = v9[18];
  if ( v9[19] != v56 )
  {
    v75 = v55;
    v57 = v9;
    v58 = 0;
    v79 = v35;
    v59 = 0;
    do
    {
      v60 = v56 + 80 * v59;
      v61 = (__int64 *)sub_159C470(v57[12], 0, 0);
      v62 = v57[12];
      v83[0] = v61;
      v63 = *(_QWORD *)&v79[8 * v59];
      v59 = (unsigned int)++v58;
      v83[1] = (__int64 *)sub_159C470(v62, v63, 0);
      v64 = *v82;
      v84[4] = 0;
      v65 = sub_15A2E80(v64, v75, v83, 2u, 1u, (__int64)v84, 0);
      v66 = *v57;
      v67 = v57[6];
      v85 = 259;
      *(_QWORD *)v84 = "bits";
      v68 = sub_15E57E0(v67, 0, 8, (__int64)v84, v65, v66);
      sub_164D160(*(_QWORD *)(v60 + 56), v68, (__m128)0LL, a3, a4, a5, v69, v70, a8, a9);
      sub_15E55B0(*(_QWORD *)(v60 + 56));
      v56 = v57[18];
      result = 0xCCCCCCCCCCCCCCCDLL * ((v57[19] - v56) >> 4);
    }
    while ( v58 != result );
    v35 = v79;
  }
  if ( src )
    result = j_j___libc_free_0(src, v88 - (_QWORD)src);
  if ( v35 )
    return j_j___libc_free_0(v35, v73);
  return result;
}
