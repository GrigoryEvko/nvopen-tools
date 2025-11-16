// Function: sub_1AF1D40
// Address: 0x1af1d40
//
void __fastcall sub_1AF1D40(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 *v13; // rdx
  __int64 v14; // rsi
  _QWORD *v15; // rsi
  char v16; // bl
  _QWORD *v17; // r13
  double v18; // xmm4_8
  double v19; // xmm5_8
  _QWORD *v20; // rax
  unsigned __int64 *v21; // r12
  _QWORD *v22; // rax
  __int64 *v23; // r14
  unsigned __int64 *v24; // r15
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  int v28; // edi
  __int64 v29; // rcx
  unsigned int v30; // r8d
  __int64 *v31; // rsi
  _QWORD *v32; // rdx
  __int64 *v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // r9
  unsigned int v36; // r8d
  __int64 *v37; // rsi
  __int64 v38; // r11
  __int64 v39; // r14
  unsigned int v40; // r8d
  __int64 *v41; // rsi
  _QWORD *v42; // r9
  __int64 v43; // r15
  __int64 v44; // rdx
  _BYTE *v45; // rax
  int v46; // r8d
  int v47; // r9d
  _BYTE *v48; // rsi
  __int64 *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 *v53; // r13
  __int64 *v54; // r14
  __int64 v55; // r15
  unsigned int v56; // r8d
  _QWORD *v57; // rdx
  _QWORD *v58; // rsi
  __int64 v59; // rax
  __int64 v60; // r8
  _BYTE *v61; // rax
  __int64 v62; // r8
  int v63; // eax
  int v64; // edx
  unsigned int v65; // r8d
  _QWORD *v66; // r12
  _QWORD *v67; // rax
  __int64 v68; // r14
  __int64 v69; // rdi
  __int64 v70; // rbx
  int v71; // r12d
  unsigned __int64 v72; // rsi
  const __m128i *v73; // rsi
  __int64 v74; // r15
  _QWORD *v75; // rax
  unsigned __int64 v76; // rcx
  __int64 v77; // r13
  int v78; // ebx
  __int64 v79; // r15
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // r8
  int v82; // r14d
  unsigned __int64 v83; // rax
  int v84; // edx
  const __m128i *v85; // rsi
  _QWORD *v86; // rax
  unsigned __int64 v87; // rax
  __int64 ***v88; // r12
  _QWORD *v89; // rax
  __int64 v90; // rax
  unsigned __int64 v91; // rax
  __int64 v92; // rax
  double v93; // xmm4_8
  double v94; // xmm5_8
  _QWORD *v95; // rax
  const __m128i *v96; // rsi
  int v97; // esi
  int v98; // r11d
  __int64 v99; // rsi
  int v100; // esi
  int v101; // r9d
  int v102; // esi
  int v103; // edx
  int v104; // r9d
  int v105; // r12d
  __int64 v106; // rsi
  _QWORD *v107; // [rsp+8h] [rbp-298h]
  unsigned __int64 v108; // [rsp+10h] [rbp-290h]
  _QWORD *v111; // [rsp+28h] [rbp-278h]
  unsigned int v112; // [rsp+30h] [rbp-270h]
  __int64 v113; // [rsp+30h] [rbp-270h]
  unsigned __int64 v114; // [rsp+38h] [rbp-268h] BYREF
  const __m128i *v115; // [rsp+40h] [rbp-260h] BYREF
  __m128i *v116; // [rsp+48h] [rbp-258h]
  const __m128i *v117; // [rsp+50h] [rbp-250h]
  __m128i v118; // [rsp+60h] [rbp-240h] BYREF
  __int64 v119; // [rsp+70h] [rbp-230h] BYREF
  int v120; // [rsp+78h] [rbp-228h]

  v114 = (unsigned __int64)a1;
  v11 = a1[6];
  if ( !v11 )
    goto LABEL_136;
  if ( *(_BYTE *)(v11 - 8) == 77 )
  {
    while ( 1 )
    {
      v12 = (_QWORD *)(v11 - 24);
      if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
        v13 = *(__int64 **)(v11 - 32);
      else
        v13 = &v12[-3 * (*(_DWORD *)(v11 - 4) & 0xFFFFFFF)];
      v14 = *v13;
      if ( *v13 && (_QWORD *)v14 == v12 )
        v14 = sub_1599EF0(*(__int64 ***)(v11 - 24));
      sub_164D160((__int64)v12, v14, a4, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, a8, a9, a10, a11);
      sub_15F20C0(v12);
      a1 = (_QWORD *)v114;
      v11 = *(_QWORD *)(v114 + 48);
      if ( !v11 )
        break;
      if ( *(_BYTE *)(v11 - 8) != 77 )
        goto LABEL_10;
    }
LABEL_136:
    BUG();
  }
LABEL_10:
  v15 = (_QWORD *)v114;
  v16 = 1;
  v115 = 0;
  v116 = 0;
  v17 = (_QWORD *)sub_157F0B0((__int64)a1);
  v117 = 0;
  v20 = *(_QWORD **)(*(_QWORD *)(v114 + 56) + 80LL);
  if ( v20 )
    v20 -= 3;
  if ( v17 == v20 || (v16 = 0, !a3) )
  {
    if ( *(_WORD *)(v114 + 18) )
      goto LABEL_100;
    goto LABEL_15;
  }
  v70 = v17[1];
  if ( v70 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v70) + 16) - 25) > 9u )
    {
      v70 = *(_QWORD *)(v70 + 8);
      if ( !v70 )
        goto LABEL_102;
    }
    v71 = 0;
    while ( 1 )
    {
      v70 = *(_QWORD *)(v70 + 8);
      if ( !v70 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v70) + 16) - 25) <= 9u )
      {
        v70 = *(_QWORD *)(v70 + 8);
        ++v71;
        if ( !v70 )
          goto LABEL_78;
      }
    }
LABEL_78:
    v72 = (unsigned int)(2 * v71 + 3);
  }
  else
  {
LABEL_102:
    v72 = 1;
  }
  sub_1953AE0(&v115, v72);
  v118.m128i_i64[0] = (__int64)v17;
  v73 = v116;
  v118.m128i_i64[1] = v114 | 4;
  if ( v116 == v117 )
  {
    sub_17F2860(&v115, v116, &v118);
  }
  else
  {
    if ( v116 )
    {
      a6 = _mm_loadu_si128(&v118);
      *v116 = a6;
      v73 = v116;
    }
    v116 = (__m128i *)&v73[1];
  }
  v74 = v17[1];
  if ( !v74 )
  {
LABEL_98:
    v15 = (_QWORD *)v114;
    v16 = 0;
    goto LABEL_99;
  }
  while ( 1 )
  {
    v75 = sub_1648700(v74);
    if ( (unsigned __int8)(*((_BYTE *)v75 + 16) - 25) <= 9u )
      break;
    v74 = *(_QWORD *)(v74 + 8);
    if ( !v74 )
      goto LABEL_98;
  }
  v76 = (unsigned __int64)v17;
  v107 = v17;
  v77 = v74;
  v113 = v76 | 4;
  do
  {
    v85 = v116;
    v118.m128i_i64[0] = v75[5];
    v118.m128i_i64[1] = v113;
    if ( v116 == v117 )
    {
      sub_17F2860(&v115, v116, &v118);
    }
    else
    {
      if ( v116 )
      {
        a4 = (__m128)_mm_loadu_si128(&v118);
        *v116 = (__m128i)a4;
        v85 = v116;
      }
      v116 = (__m128i *)&v85[1];
    }
    v86 = sub_1648700(v77);
    v87 = sub_157EBA0(v86[5]);
    v81 = v87;
    if ( v87 )
    {
      v78 = sub_15F4D60(v87);
      v79 = sub_1648700(v77)[5];
      v80 = sub_157EBA0(v79);
      v81 = v80;
      if ( v80 )
      {
        v108 = v80;
        v82 = sub_15F4D60(v80);
        v83 = sub_157EBA0(v79);
        v81 = v108;
      }
      else
      {
        v83 = 0;
        v82 = 0;
      }
    }
    else
    {
      v83 = 0;
      v78 = 0;
      v82 = 0;
    }
    v118.m128i_i64[0] = v83;
    v118.m128i_i32[2] = 0;
    v119 = v81;
    v120 = v82;
    sub_1AED640((__int64)&v118, &v114);
    if ( v78 != v84 )
      break;
    v95 = sub_1648700(v77);
    v96 = v116;
    v118.m128i_i64[0] = v95[5];
    v118.m128i_i64[1] = v114 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v116 == v117 )
    {
      sub_17F2860(&v115, v116, &v118);
      break;
    }
    if ( v116 )
    {
      a5 = _mm_loadu_si128(&v118);
      *v116 = a5;
      v96 = v116;
    }
    v116 = (__m128i *)&v96[1];
    v77 = *(_QWORD *)(v77 + 8);
    if ( !v77 )
      goto LABEL_107;
LABEL_90:
    v75 = sub_1648700(v77);
  }
  while ( (unsigned __int8)(*((_BYTE *)v75 + 16) - 25) <= 9u );
  v77 = *(_QWORD *)(v77 + 8);
  if ( v77 )
    goto LABEL_90;
LABEL_107:
  v17 = v107;
  v15 = (_QWORD *)v114;
  v16 = 0;
LABEL_99:
  if ( *((_WORD *)v15 + 9) )
  {
LABEL_100:
    v88 = (__int64 ***)sub_159BF40((__int64)v15);
    v89 = (_QWORD *)sub_16498A0((__int64)v88);
    v90 = sub_1643350(v89);
    v91 = sub_159C470(v90, 1, 0);
    v92 = sub_15A3BA0(v91, *v88, 0);
    sub_164D160((__int64)v88, v92, a4, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, v93, v94, a10, a11);
    sub_159D850((__int64)v88);
    v15 = (_QWORD *)v114;
  }
LABEL_15:
  v21 = v17 + 5;
  sub_164D160((__int64)v17, (__int64)v15, a4, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, v18, v19, a10, a11);
  v22 = (_QWORD *)sub_157EBA0((__int64)v17);
  sub_15F20C0(v22);
  if ( v17 + 5 != (_QWORD *)(v17[5] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v23 = (__int64 *)v17[6];
    v24 = *(unsigned __int64 **)(v114 + 48);
    if ( v21 != v24 )
    {
      if ( (unsigned __int64 *)(v114 + 40) != v21 )
        sub_157EA80(v114 + 40, (__int64)(v17 + 5), v17[6], (__int64)(v17 + 5));
      if ( v21 != v24 && v21 != (unsigned __int64 *)v23 )
      {
        v25 = v17[5] & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v21;
        v17[5] = v17[5] & 7LL | *v23 & 0xFFFFFFFFFFFFFFF8LL;
        v26 = *v24;
        *(_QWORD *)(v25 + 8) = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        *v23 = v26 | *v23 & 7;
        *(_QWORD *)(v26 + 8) = v23;
        *v24 = v25 | *v24 & 7;
      }
    }
  }
  if ( v16 )
    sub_1580AC0((_QWORD *)v114, (__int64)v17);
  if ( !a2 )
    goto LABEL_64;
  v27 = *(unsigned int *)(a2 + 48);
  if ( !(_DWORD)v27 )
    goto LABEL_64;
  v28 = v27 - 1;
  v29 = *(_QWORD *)(a2 + 32);
  v112 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
  v30 = (v27 - 1) & v112;
  v31 = (__int64 *)(v29 + 16LL * v30);
  v32 = (_QWORD *)*v31;
  if ( v17 != (_QWORD *)*v31 )
  {
    v100 = 1;
    while ( v32 != (_QWORD *)-8LL )
    {
      v101 = v100 + 1;
      v30 = v28 & (v100 + v30);
      v31 = (__int64 *)(v29 + 16LL * v30);
      v32 = (_QWORD *)*v31;
      if ( v17 == (_QWORD *)*v31 )
        goto LABEL_27;
      v100 = v101;
    }
    goto LABEL_64;
  }
LABEL_27:
  v33 = (__int64 *)(v29 + 16LL * (unsigned int)v27);
  if ( v33 == v31 )
    goto LABEL_64;
  v34 = v31[1];
  if ( !v34 )
    goto LABEL_64;
  v35 = **(_QWORD **)(v34 + 8);
  v36 = v28 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v37 = (__int64 *)(v29 + 16LL * v36);
  v38 = *v37;
  if ( v35 == *v37 )
  {
LABEL_30:
    if ( v37 != v33 )
    {
      v39 = v37[1];
      goto LABEL_32;
    }
  }
  else
  {
    v102 = 1;
    while ( v38 != -8 )
    {
      v105 = v102 + 1;
      v106 = v28 & (v36 + v102);
      v36 = v106;
      v37 = (__int64 *)(v29 + 16 * v106);
      v38 = *v37;
      if ( v35 == *v37 )
        goto LABEL_30;
      v102 = v105;
    }
  }
  v39 = 0;
LABEL_32:
  v40 = v28 & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
  v41 = (__int64 *)(v29 + 16LL * v40);
  v42 = (_QWORD *)*v41;
  if ( v114 != *v41 )
  {
    v97 = 1;
    while ( v42 != (_QWORD *)-8LL )
    {
      v98 = v97 + 1;
      v99 = v28 & (v40 + v97);
      v40 = v99;
      v41 = (__int64 *)(v29 + 16 * v99);
      v42 = (_QWORD *)*v41;
      if ( v114 == *v41 )
        goto LABEL_33;
      v97 = v98;
    }
LABEL_135:
    *(_BYTE *)(a2 + 72) = 0;
    BUG();
  }
LABEL_33:
  if ( v41 == v33 )
    goto LABEL_135;
  v43 = v41[1];
  *(_BYTE *)(a2 + 72) = 0;
  v44 = *(_QWORD *)(v43 + 8);
  if ( v44 == v39 )
    goto LABEL_53;
  v118.m128i_i64[0] = v43;
  v45 = sub_1AE76A0(*(_QWORD **)(v44 + 24), *(_QWORD *)(v44 + 32), v118.m128i_i64);
  sub_15CDF70(*(_QWORD *)(v43 + 8) + 24LL, v45);
  *(_QWORD *)(v43 + 8) = v39;
  v118.m128i_i64[0] = v43;
  v48 = *(_BYTE **)(v39 + 32);
  if ( v48 == *(_BYTE **)(v39 + 40) )
  {
    sub_15CE310(v39 + 24, v48, &v118);
  }
  else
  {
    if ( v48 )
    {
      *(_QWORD *)v48 = v43;
      v48 = *(_BYTE **)(v39 + 32);
    }
    *(_QWORD *)(v39 + 32) = v48 + 8;
  }
  if ( *(_DWORD *)(v43 + 16) != *(_DWORD *)(*(_QWORD *)(v43 + 8) + 16LL) + 1 )
  {
    v118.m128i_i64[0] = (__int64)&v119;
    v49 = &v119;
    v119 = v43;
    v111 = v17;
    v118.m128i_i64[1] = 0x4000000001LL;
    LODWORD(v50) = 1;
    do
    {
      v51 = (unsigned int)v50;
      v50 = (unsigned int)(v50 - 1);
      v52 = v49[v51 - 1];
      v118.m128i_i32[2] = v50;
      v53 = *(__int64 **)(v52 + 32);
      v54 = *(__int64 **)(v52 + 24);
      *(_DWORD *)(v52 + 16) = *(_DWORD *)(*(_QWORD *)(v52 + 8) + 16LL) + 1;
      if ( v54 != v53 )
      {
        do
        {
          v55 = *v54;
          if ( *(_DWORD *)(*v54 + 16) != *(_DWORD *)(*(_QWORD *)(*v54 + 8) + 16LL) + 1 )
          {
            if ( v118.m128i_i32[3] <= (unsigned int)v50 )
            {
              sub_16CD150((__int64)&v118, &v119, 0, 8, v46, v47);
              v50 = v118.m128i_u32[2];
            }
            *(_QWORD *)(v118.m128i_i64[0] + 8 * v50) = v55;
            v50 = (unsigned int)++v118.m128i_i32[2];
          }
          ++v54;
        }
        while ( v53 != v54 );
        v49 = (__int64 *)v118.m128i_i64[0];
      }
    }
    while ( (_DWORD)v50 );
    v17 = v111;
    if ( v49 != &v119 )
      _libc_free((unsigned __int64)v49);
  }
  v29 = *(_QWORD *)(a2 + 32);
  v27 = *(unsigned int *)(a2 + 48);
  if ( !(_DWORD)v27 )
  {
LABEL_134:
    v118.m128i_i64[0] = 0;
    *(_BYTE *)(a2 + 72) = 0;
    BUG();
  }
  v28 = v27 - 1;
LABEL_53:
  v56 = v28 & v112;
  v57 = (_QWORD *)(v29 + 16LL * (v28 & v112));
  v58 = (_QWORD *)*v57;
  if ( v17 != (_QWORD *)*v57 )
  {
    v103 = 1;
    while ( v58 != (_QWORD *)-8LL )
    {
      v104 = v103 + 1;
      v56 = v28 & (v103 + v56);
      v57 = (_QWORD *)(v29 + 16LL * v56);
      v58 = (_QWORD *)*v57;
      if ( v17 == (_QWORD *)*v57 )
        goto LABEL_54;
      v103 = v104;
    }
    goto LABEL_134;
  }
LABEL_54:
  if ( v57 == (_QWORD *)(v29 + 16 * v27) )
    goto LABEL_134;
  v118.m128i_i64[0] = v57[1];
  v59 = v118.m128i_i64[0];
  *(_BYTE *)(a2 + 72) = 0;
  v60 = *(_QWORD *)(v59 + 8);
  if ( v60 )
  {
    v61 = sub_1AE76A0(*(_QWORD **)(v60 + 24), *(_QWORD *)(v60 + 32), v118.m128i_i64);
    sub_15CDF70(v62 + 24, v61);
    v63 = *(_DWORD *)(a2 + 48);
    v29 = *(_QWORD *)(a2 + 32);
    if ( v63 )
    {
      v28 = v63 - 1;
      goto LABEL_58;
    }
  }
  else
  {
LABEL_58:
    v64 = 1;
    v65 = v28 & v112;
    v66 = (_QWORD *)(v29 + 16LL * (v28 & v112));
    v67 = (_QWORD *)*v66;
    if ( v17 == (_QWORD *)*v66 )
    {
LABEL_59:
      v68 = v66[1];
      if ( v68 )
      {
        v69 = *(_QWORD *)(v68 + 24);
        if ( v69 )
          j_j___libc_free_0(v69, *(_QWORD *)(v68 + 40) - v69);
        j_j___libc_free_0(v68, 56);
      }
      *v66 = -16;
      --*(_DWORD *)(a2 + 40);
      ++*(_DWORD *)(a2 + 44);
    }
    else
    {
      while ( v67 != (_QWORD *)-8LL )
      {
        v65 = v28 & (v64 + v65);
        v66 = (_QWORD *)(v29 + 16LL * v65);
        v67 = (_QWORD *)*v66;
        if ( v17 == (_QWORD *)*v66 )
          goto LABEL_59;
        ++v64;
      }
    }
  }
LABEL_64:
  if ( a3 )
  {
    sub_15CD5A0(a3, (__int64)v17);
    if ( v16 )
      sub_15D3960(a3, *(_QWORD *)(v114 + 56));
    else
      sub_15CD9D0(a3, v115->m128i_i64, v116 - v115);
  }
  else
  {
    sub_157F980((__int64)v17);
  }
  if ( v115 )
    j_j___libc_free_0(v115, (char *)v117 - (char *)v115);
}
