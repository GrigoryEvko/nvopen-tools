// Function: sub_1BAF4D0
// Address: 0x1baf4d0
//
__int64 __fastcall sub_1BAF4D0(
        __int64 a1,
        __m128 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 *v12; // r12
  __int64 v13; // r9
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // edx
  __int64 v21; // rsi
  __int64 v22; // rdi
  int v23; // edx
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // r14
  int v32; // edx
  int v33; // edx
  __int64 v34; // r9
  int v35; // eax
  unsigned int v36; // ecx
  __int64 *v37; // rdi
  __int64 v38; // r8
  int v39; // eax
  __int64 v40; // rax
  __int64 *v41; // r12
  __int64 *v42; // r13
  __int64 v43; // rsi
  char *v44; // rax
  __int64 v45; // r13
  __int64 v46; // r15
  __int64 result; // rax
  char v48; // cl
  _QWORD *v49; // r13
  int v50; // ebx
  __int64 v51; // rax
  __int64 *v52; // rdi
  __int64 *v53; // rsi
  int v54; // eax
  double v55; // xmm4_8
  double v56; // xmm5_8
  __int64 v57; // rcx
  unsigned __int64 *v58; // rdx
  int v59; // r11d
  int v60; // r12d
  bool v61; // r10
  bool v62; // zf
  _QWORD *v63; // r14
  unsigned int v64; // r13d
  bool v65; // r10
  __int64 *v66; // rbx
  __int64 v67; // rsi
  char v68; // al
  __int64 v69; // rsi
  __int64 **v70; // r12
  __int64 v71; // r12
  __int64 v72; // r14
  char v73; // al
  _QWORD *v74; // rdx
  unsigned int v75; // eax
  unsigned int v76; // esi
  __int64 *v77; // rax
  int v78; // r11d
  int v79; // eax
  int v80; // eax
  int v81; // r8d
  __int64 *v82; // r15
  unsigned int v83; // r13d
  int v84; // r11d
  __int64 *v85; // r10
  unsigned __int64 *v86; // [rsp+0h] [rbp-D0h]
  int v87; // [rsp+8h] [rbp-C8h]
  __int64 v88; // [rsp+10h] [rbp-C0h]
  __int64 v89; // [rsp+28h] [rbp-A8h]
  __int64 v90; // [rsp+28h] [rbp-A8h]
  int v91; // [rsp+3Ch] [rbp-94h] BYREF
  __int64 *v92; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v93; // [rsp+48h] [rbp-88h] BYREF
  __int64 *v94; // [rsp+50h] [rbp-80h] BYREF
  __int64 v95; // [rsp+58h] [rbp-78h]
  _QWORD *v96; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v97; // [rsp+68h] [rbp-68h]
  char v98; // [rsp+A0h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a1 + 88) > 1u )
    sub_1B9C6F0(a1, a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, a6, a7, a8, a9);
  sub_1BA5420(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  sub_1BAD1B0(a1);
  v10 = *(_QWORD *)(a1 + 448);
  v89 = a1 + 472;
  v11 = *(__int64 **)(v10 + 144);
  v12 = *(__int64 **)(v10 + 136);
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      v30 = *(_DWORD *)(a1 + 496);
      v31 = *(_QWORD *)(a1 + 184);
      if ( !v30 )
        break;
      v13 = *(_QWORD *)(a1 + 480);
      v14 = (v30 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( *v12 != *v15 )
      {
        v78 = 1;
        v37 = 0;
        while ( v16 != -8 )
        {
          if ( v16 != -16 || v37 )
            v15 = v37;
          v14 = (v30 - 1) & (v78 + v14);
          v82 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v82;
          if ( *v12 == *v82 )
          {
            v17 = v82[1];
            goto LABEL_7;
          }
          ++v78;
          v37 = v15;
          v15 = (__int64 *)(v13 + 16LL * v14);
        }
        if ( !v37 )
          v37 = v15;
        v79 = *(_DWORD *)(a1 + 488);
        ++*(_QWORD *)(a1 + 472);
        v39 = v79 + 1;
        if ( 4 * v39 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(a1 + 492) - v39 > v30 >> 3 )
            goto LABEL_16;
          sub_1BA3880(v89, v30);
          sub_1BA0D30(v89, v12, &v94);
          v37 = v94;
          v35 = *(_DWORD *)(a1 + 488);
LABEL_15:
          v39 = v35 + 1;
LABEL_16:
          *(_DWORD *)(a1 + 488) = v39;
          if ( *v37 != -8 )
            --*(_DWORD *)(a1 + 492);
          v40 = *v12;
          v17 = 0;
          v37[1] = 0;
          *v37 = v40;
          goto LABEL_7;
        }
LABEL_13:
        sub_1BA3880(v89, 2 * v30);
        v32 = *(_DWORD *)(a1 + 496);
        if ( !v32 )
        {
          ++*(_DWORD *)(a1 + 488);
          BUG();
        }
        v33 = v32 - 1;
        v34 = *(_QWORD *)(a1 + 480);
        v35 = *(_DWORD *)(a1 + 488);
        v36 = v33 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v37 = (__int64 *)(v34 + 16LL * v36);
        v38 = *v37;
        if ( *v12 == *v37 )
          goto LABEL_15;
        v84 = 1;
        v85 = 0;
        while ( v38 != -8 )
        {
          if ( v38 != -16 || v85 )
            v37 = v85;
          v36 = v33 & (v84 + v36);
          v38 = *(_QWORD *)(v34 + 16LL * v36);
          if ( *v12 == v38 )
          {
            v39 = v35 + 1;
            v37 = (__int64 *)(v34 + 16LL * v36);
            goto LABEL_16;
          }
          ++v84;
          v85 = v37;
          v37 = (__int64 *)(v34 + 16LL * v36);
        }
        v39 = v35 + 1;
        if ( v85 )
          v37 = v85;
        goto LABEL_16;
      }
      v17 = v15[1];
LABEL_7:
      v18 = *(_QWORD *)(a1 + 24);
      v19 = 0;
      v20 = *(_DWORD *)(v18 + 24);
      if ( v20 )
      {
        v21 = *(_QWORD *)(a1 + 200);
        v22 = *(_QWORD *)(v18 + 8);
        v23 = v20 - 1;
        v24 = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v25 = (__int64 *)(v22 + 16LL * v24);
        v26 = *v25;
        if ( v21 == *v25 )
        {
LABEL_9:
          v19 = v25[1];
        }
        else
        {
          v80 = 1;
          while ( v26 != -8 )
          {
            v81 = v80 + 1;
            v24 = v23 & (v80 + v24);
            v25 = (__int64 *)(v22 + 16LL * v24);
            v26 = *v25;
            if ( v21 == *v25 )
              goto LABEL_9;
            v80 = v81;
          }
          v19 = 0;
        }
      }
      v27 = sub_1B97190(a1, v19, (__m128i)a2, a3, *(double *)a4.m128i_i64);
      v28 = *v12;
      v29 = (__int64)(v12 + 1);
      v12 += 11;
      sub_1BA3A10(a1, v28, v29, v27, v17, v31, (__m128i)a2, a3, *(double *)a4.m128i_i64);
      if ( v11 == v12 )
        goto LABEL_19;
    }
    ++*(_QWORD *)(a1 + 472);
    goto LABEL_13;
  }
LABEL_19:
  sub_1B91F40(a1);
  v41 = *(__int64 **)(a1 + 384);
  v42 = &v41[*(unsigned int *)(a1 + 392)];
  while ( v42 != v41 )
  {
    v43 = *v41++;
    sub_1BAEFD0(a1, v43);
  }
  v44 = (char *)&v96;
  v94 = 0;
  v45 = *(_QWORD *)(a1 + 200);
  v95 = 1;
  do
  {
    *(_QWORD *)v44 = -8;
    v44 += 16;
  }
  while ( v44 != &v98 );
  v46 = *(_QWORD *)(v45 + 48);
  result = v45 + 40;
  v48 = v95;
  v90 = v45 + 40;
  if ( v46 == v45 + 40 )
    goto LABEL_51;
  while ( 1 )
  {
LABEL_35:
    v71 = v46;
    v46 = *(_QWORD *)(v46 + 8);
    v72 = v71 - 24;
    v92 = (__int64 *)(v71 - 24);
    result = *(unsigned __int8 *)(v71 - 8);
    if ( (unsigned __int8)(result - 83) > 2u && (_BYTE)result != 56 )
      goto LABEL_34;
    if ( (v48 & 1) != 0 )
    {
      v49 = &v96;
      v50 = 3;
      goto LABEL_26;
    }
    v49 = v96;
    if ( v97 )
      break;
    v70 = (__int64 **)&v93;
LABEL_43:
    v73 = sub_1B98860((__int64)&v94, &v92, v70);
    v74 = (_QWORD *)v93;
    if ( v73 )
      goto LABEL_50;
    v94 = (__int64 *)((char *)v94 + 1);
    v75 = ((unsigned int)v95 >> 1) + 1;
    if ( (v95 & 1) != 0 )
    {
      v76 = 4;
      if ( 4 * v75 >= 0xC )
      {
LABEL_67:
        v76 *= 2;
LABEL_68:
        sub_1B989E0((__int64)&v94, v76);
        sub_1B98860((__int64)&v94, &v92, v70);
        v74 = (_QWORD *)v93;
        v75 = ((unsigned int)v95 >> 1) + 1;
        goto LABEL_47;
      }
    }
    else
    {
      v76 = v97;
      if ( 3 * v97 <= 4 * v75 )
        goto LABEL_67;
    }
    if ( v76 - (v75 + HIDWORD(v95)) <= v76 >> 3 )
      goto LABEL_68;
LABEL_47:
    LODWORD(v95) = v95 & 1 | (2 * v75);
    if ( *v74 != -8 )
      --HIDWORD(v95);
    v77 = v92;
    v74[1] = 0;
    *v74 = v77;
LABEL_50:
    result = (__int64)v92;
    v74[1] = v92;
    v48 = v95;
    if ( v90 == v46 )
      goto LABEL_51;
  }
  v50 = v97 - 1;
LABEL_26:
  v51 = 3LL * (*(_DWORD *)(v71 - 4) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v71 - 1) & 0x40) != 0 )
  {
    v52 = *(__int64 **)(v71 - 32);
    v53 = &v52[v51];
  }
  else
  {
    v53 = (__int64 *)(v71 - 24);
    v52 = (__int64 *)(v72 - v51 * 8);
  }
  v93 = sub_1B98460(v52, v53);
  v91 = *(unsigned __int8 *)(v71 - 8) - 24;
  v54 = sub_18FDAA0(&v91, (__int64 *)&v93);
  v57 = v71 - 24;
  v58 = &v93;
  v59 = 1;
  v60 = v50;
  v61 = v72 == -16;
  v62 = v72 == -8;
  v63 = v49;
  v64 = v50 & v54;
  v65 = v62 || v61;
  while ( 2 )
  {
    v66 = &v63[2 * v64];
    v67 = *v66;
    if ( *v66 != -16 && *v66 != -8 && !v65 )
    {
      v86 = v58;
      v87 = v59;
      v88 = v57;
      v68 = sub_15F41F0(v57, v67);
      v57 = v88;
      v65 = 0;
      v59 = v87;
      v58 = v86;
      if ( v68 )
        break;
      if ( *v66 == -8 )
      {
LABEL_42:
        v70 = (__int64 **)v58;
        goto LABEL_43;
      }
      goto LABEL_75;
    }
    if ( v57 != v67 )
    {
      if ( v67 == -8 )
        goto LABEL_42;
LABEL_75:
      v83 = v59 + v64;
      ++v59;
      v64 = v60 & v83;
      continue;
    }
    break;
  }
  v69 = v66[1];
  v70 = (__int64 **)v58;
  if ( !v69 )
    goto LABEL_43;
  sub_164D160((__int64)v92, v69, a2, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v55, v56, a8, a9);
  result = sub_15F20C0(v92);
  v48 = v95;
LABEL_34:
  if ( v90 != v46 )
    goto LABEL_35;
LABEL_51:
  if ( (v48 & 1) == 0 )
    return j___libc_free_0(v96);
  return result;
}
