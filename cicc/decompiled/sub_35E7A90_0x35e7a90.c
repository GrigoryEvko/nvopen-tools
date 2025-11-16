// Function: sub_35E7A90
// Address: 0x35e7a90
//
__int64 __fastcall sub_35E7A90(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rax
  int *v3; // r13
  char *v4; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 *v8; // rsi
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // r11d
  __int64 *v12; // rdx
  unsigned int v13; // esi
  __int64 *v14; // rcx
  __int64 *v15; // rdi
  _DWORD *v16; // rcx
  int v17; // r12d
  __int64 v18; // rcx
  int v19; // r11d
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 *v23; // r8
  __int64 v24; // rdx
  int v25; // r12d
  __int64 v26; // r8
  int v27; // ecx
  unsigned int v28; // esi
  __int64 *v29; // r9
  int v30; // r14d
  unsigned int v31; // esi
  int v32; // eax
  __int64 *v33; // rsi
  __int64 *v34; // r13
  signed __int64 v35; // r11
  unsigned __int64 v36; // rdx
  int v37; // r10d
  unsigned int v38; // r14d
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rax
  char *v44; // rdi
  char *v45; // r11
  int v46; // esi
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 (*v49)(void); // rdx
  __int64 v50; // rax
  unsigned int v52; // edx
  __int64 *v53; // r8
  int v54; // r14d
  __int64 v55; // r9
  __int64 v56; // r9
  int v57; // r14d
  unsigned int v58; // edx
  _QWORD *v59; // rcx
  _QWORD *v60; // r9
  _QWORD *v61; // rax
  int v62; // esi
  __int64 **v63; // r14
  int v64; // r14d
  char *v65; // [rsp+8h] [rbp-1D8h]
  size_t n; // [rsp+10h] [rbp-1D0h]
  int na; // [rsp+10h] [rbp-1D0h]
  int v68; // [rsp+20h] [rbp-1C0h]
  __int64 v69; // [rsp+24h] [rbp-1BCh]
  int v70; // [rsp+2Ch] [rbp-1B4h]
  __int64 v72; // [rsp+30h] [rbp-1B0h]
  unsigned int v73; // [rsp+38h] [rbp-1A8h]
  void *src; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 *v75; // [rsp+48h] [rbp-198h]
  __int64 *v76; // [rsp+50h] [rbp-190h]
  __int64 v77; // [rsp+60h] [rbp-180h] BYREF
  __int64 v78; // [rsp+68h] [rbp-178h]
  __int64 v79; // [rsp+70h] [rbp-170h]
  unsigned int v80; // [rsp+78h] [rbp-168h]
  __int64 v81; // [rsp+80h] [rbp-160h] BYREF
  __int64 v82; // [rsp+88h] [rbp-158h]
  __int64 v83; // [rsp+90h] [rbp-150h]
  unsigned int v84; // [rsp+98h] [rbp-148h]
  __int64 v85; // [rsp+A0h] [rbp-140h] BYREF
  unsigned __int64 v86; // [rsp+A8h] [rbp-138h]
  char *v87; // [rsp+B0h] [rbp-130h]
  char *v88; // [rsp+B8h] [rbp-128h]
  __int64 v89; // [rsp+C0h] [rbp-120h]
  __int64 v90; // [rsp+C8h] [rbp-118h]
  __int64 v91; // [rsp+D0h] [rbp-110h]
  unsigned int v92; // [rsp+D8h] [rbp-108h]
  __int64 v93; // [rsp+E0h] [rbp-100h]
  __int64 v94; // [rsp+E8h] [rbp-F8h]
  int v95; // [rsp+F0h] [rbp-F0h]
  int v96; // [rsp+F4h] [rbp-ECh]
  unsigned int v97; // [rsp+F8h] [rbp-E8h]
  int v98; // [rsp+100h] [rbp-E0h]
  __int64 *v99; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v100; // [rsp+118h] [rbp-C8h]
  unsigned __int64 v101; // [rsp+120h] [rbp-C0h]
  __int64 v102; // [rsp+128h] [rbp-B8h]
  __int64 v103; // [rsp+130h] [rbp-B0h]
  __int64 v104; // [rsp+138h] [rbp-A8h]
  __int64 v105; // [rsp+140h] [rbp-A0h]
  __int64 v106; // [rsp+148h] [rbp-98h]
  __int64 v107; // [rsp+150h] [rbp-90h]
  __int64 v108; // [rsp+158h] [rbp-88h]
  int v109; // [rsp+168h] [rbp-78h] BYREF
  unsigned __int64 v110; // [rsp+170h] [rbp-70h]
  int *v111; // [rsp+178h] [rbp-68h]
  int *v112; // [rsp+180h] [rbp-60h]
  __int64 v113; // [rsp+188h] [rbp-58h]
  __int64 v114; // [rsp+190h] [rbp-50h]
  __int64 v115; // [rsp+198h] [rbp-48h]
  __int64 v116; // [rsp+1A0h] [rbp-40h]
  unsigned int v117; // [rsp+1A8h] [rbp-38h]

  v1 = (_QWORD *)a1;
  v2 = *(unsigned int *)(a1 + 280);
  v3 = *(int **)(a1 + 272);
  v4 = (char *)&v3[6 * v2];
  sub_35E79A0((__int64 *)&v99, v3, 0xAAAAAAAAAAAAAAABLL * ((24 * v2) >> 3));
  if ( v101 )
    sub_35E6830((char *)v3, v4, v101, v100);
  else
    sub_35E5FE0((char *)v3, v4);
  j_j___libc_free_0(v101);
  v5 = *(unsigned int *)(a1 + 280);
  v6 = *(_QWORD *)(a1 + 272);
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v7 = v6 + 24 * v5;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  src = 0;
  v75 = 0;
  v76 = 0;
  if ( v6 == v7 )
  {
    v70 = 0;
    v41 = 0;
    v36 = 0;
    v33 = 0;
    v73 = 0;
    v35 = 0;
    v38 = 0;
    v37 = 0;
    v40 = 1;
    v39 = 1;
    v34 = 0;
    v69 = 0;
    v72 = 0;
    goto LABEL_48;
  }
  v8 = 0;
  v9 = 0;
  while ( 1 )
  {
    v24 = *(_QWORD *)(v6 + 16);
    v99 = (__int64 *)v24;
    if ( v8 == v9 )
    {
      sub_2E997F0((__int64)&src, v8, &v99);
    }
    else
    {
      if ( v9 )
      {
        *v9 = v24;
        v9 = v75;
      }
      v75 = v9 + 1;
    }
    v25 = *(_DWORD *)(v6 + 8);
    if ( !v80 )
    {
      ++v77;
      goto LABEL_17;
    }
    v10 = (__int64)v99;
    v11 = 1;
    v12 = 0;
    v13 = (v80 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
    v14 = (__int64 *)(v78 + 16LL * v13);
    v15 = (__int64 *)*v14;
    if ( v99 == (__int64 *)*v14 )
      goto LABEL_6;
    while ( 1 )
    {
      if ( v15 == (__int64 *)-4096LL )
      {
        if ( !v12 )
          v12 = v14;
        ++v77;
        v27 = v79 + 1;
        if ( 4 * ((int)v79 + 1) < 3 * v80 )
        {
          if ( v80 - HIDWORD(v79) - v27 > v80 >> 3 )
            goto LABEL_19;
          sub_354C5D0((__int64)&v77, v80);
          if ( v80 )
          {
            v26 = (__int64)v99;
            v29 = 0;
            v30 = 1;
            v27 = v79 + 1;
            v31 = (v80 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
            v12 = (__int64 *)(v78 + 16LL * v31);
            v10 = *v12;
            if ( v99 != (__int64 *)*v12 )
            {
              while ( v10 != -4096 )
              {
                if ( !v29 && v10 == -8192 )
                  v29 = v12;
                v31 = (v80 - 1) & (v30 + v31);
                v12 = (__int64 *)(v78 + 16LL * v31);
                v10 = *v12;
                if ( v99 == (__int64 *)*v12 )
                  goto LABEL_19;
                ++v30;
              }
LABEL_31:
              v10 = v26;
              if ( v29 )
                v12 = v29;
            }
LABEL_19:
            LODWORD(v79) = v27;
            if ( *v12 != -4096 )
              --HIDWORD(v79);
            *v12 = v10;
            v16 = v12 + 1;
            *((_DWORD *)v12 + 2) = 0;
            goto LABEL_7;
          }
LABEL_128:
          LODWORD(v79) = v79 + 1;
          BUG();
        }
LABEL_17:
        sub_354C5D0((__int64)&v77, 2 * v80);
        if ( v80 )
        {
          v26 = (__int64)v99;
          v27 = v79 + 1;
          v28 = (v80 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
          v12 = (__int64 *)(v78 + 16LL * v28);
          v10 = *v12;
          if ( v99 != (__int64 *)*v12 )
          {
            v64 = 1;
            v29 = 0;
            while ( v10 != -4096 )
            {
              if ( v10 == -8192 && !v29 )
                v29 = v12;
              v28 = (v80 - 1) & (v64 + v28);
              v12 = (__int64 *)(v78 + 16LL * v28);
              v10 = *v12;
              if ( v99 == (__int64 *)*v12 )
                goto LABEL_19;
              ++v64;
            }
            goto LABEL_31;
          }
          goto LABEL_19;
        }
        goto LABEL_128;
      }
      if ( v12 || v15 != (__int64 *)-8192LL )
        v14 = v12;
      v13 = (v80 - 1) & (v11 + v13);
      v63 = (__int64 **)(v78 + 16LL * v13);
      v15 = *v63;
      if ( v99 == *v63 )
        break;
      ++v11;
      v12 = v14;
      v14 = (__int64 *)(v78 + 16LL * v13);
    }
    v14 = (__int64 *)(v78 + 16LL * v13);
LABEL_6:
    v16 = v14 + 1;
LABEL_7:
    *v16 = v25;
    v17 = *(_DWORD *)(v6 + 4);
    if ( !v84 )
    {
      ++v81;
LABEL_69:
      sub_354C5D0((__int64)&v81, 2 * v84);
      if ( !v84 )
        goto LABEL_129;
      v18 = (__int64)v99;
      v52 = (v84 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
      v32 = v83 + 1;
      v20 = v82 + 16LL * v52;
      v53 = *(__int64 **)v20;
      if ( *(__int64 **)v20 != v99 )
      {
        v54 = 1;
        v55 = 0;
        while ( v53 != (__int64 *)-4096LL )
        {
          if ( v53 == (__int64 *)-8192LL && !v55 )
            v55 = v20;
          v52 = (v84 - 1) & (v54 + v52);
          v20 = v82 + 16LL * v52;
          v53 = *(__int64 **)v20;
          if ( v99 == *(__int64 **)v20 )
            goto LABEL_44;
          ++v54;
        }
        if ( v55 )
          v20 = v55;
      }
      goto LABEL_44;
    }
    v18 = (__int64)v99;
    v19 = 1;
    v20 = 0;
    v21 = (v84 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
    v22 = v82 + 16LL * v21;
    v23 = *(__int64 **)v22;
    if ( v99 == *(__int64 **)v22 )
      break;
    while ( v23 != (__int64 *)-4096LL )
    {
      if ( !v20 && v23 == (__int64 *)-8192LL )
        v20 = v22;
      v21 = (v84 - 1) & (v19 + v21);
      v22 = v82 + 16LL * v21;
      v23 = *(__int64 **)v22;
      if ( v99 == *(__int64 **)v22 )
        goto LABEL_9;
      ++v19;
    }
    if ( !v20 )
      v20 = v22;
    ++v81;
    v32 = v83 + 1;
    if ( 4 * ((int)v83 + 1) >= 3 * v84 )
      goto LABEL_69;
    if ( v84 - HIDWORD(v83) - v32 <= v84 >> 3 )
    {
      sub_354C5D0((__int64)&v81, v84);
      if ( !v84 )
      {
LABEL_129:
        LODWORD(v83) = v83 + 1;
        BUG();
      }
      v56 = 0;
      v57 = 1;
      v58 = (v84 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
      v32 = v83 + 1;
      v20 = v82 + 16LL * v58;
      v18 = *(_QWORD *)v20;
      if ( v99 != *(__int64 **)v20 )
      {
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v56 )
            v56 = v20;
          v58 = (v84 - 1) & (v57 + v58);
          v20 = v82 + 16LL * v58;
          v18 = *(_QWORD *)v20;
          if ( v99 == *(__int64 **)v20 )
            goto LABEL_44;
          ++v57;
        }
        v18 = (__int64)v99;
        if ( v56 )
          v20 = v56;
      }
    }
LABEL_44:
    LODWORD(v83) = v32;
    if ( *(_QWORD *)v20 != -4096 )
      --HIDWORD(v83);
    v6 += 24;
    *(_QWORD *)v20 = v18;
    *(_DWORD *)(v20 + 8) = 0;
    *(_DWORD *)(v20 + 8) = v17;
    if ( v7 == v6 )
      goto LABEL_47;
LABEL_10:
    v9 = v75;
    v8 = v76;
  }
LABEL_9:
  v6 += 24;
  *(_DWORD *)(v22 + 8) = v17;
  if ( v7 != v6 )
    goto LABEL_10;
LABEL_47:
  v1 = (_QWORD *)a1;
  v33 = v75;
  v72 = v78;
  v34 = (__int64 *)src;
  v69 = v79;
  v70 = HIDWORD(v83);
  v35 = (char *)v75 - (_BYTE *)src;
  v36 = (char *)v75 - (_BYTE *)src;
  v37 = v83;
  v38 = v84;
  v39 = v77 + 1;
  v73 = v80;
  v40 = v81 + 1;
  v41 = v82;
LABEL_48:
  v77 = v39;
  v42 = v1[4];
  v81 = v40;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v76 = 0;
  v75 = 0;
  src = 0;
  v85 = v42;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  if ( v35 )
  {
    if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v40, v33, v36);
    v68 = v37;
    n = v36;
    v43 = sub_22077B0(v36);
    v36 = n;
    v37 = v68;
    v44 = (char *)v43;
  }
  else
  {
    v44 = 0;
  }
  v45 = &v44[v36];
  v86 = (unsigned __int64)v44;
  v87 = v44;
  v88 = &v44[v36];
  if ( v33 != v34 )
  {
    v65 = &v44[v36];
    na = v37;
    memmove(v44, v34, v36);
    v45 = v65;
    v37 = na;
  }
  v87 = v45;
  v89 = 1;
  v90 = v72;
  v93 = 1;
  v91 = v69;
  v94 = v41;
  v95 = v37;
  v92 = v73;
  v97 = v38;
  v96 = v70;
  v98 = 0;
  if ( !v37 )
    goto LABEL_54;
  v59 = (_QWORD *)(v41 + 16LL * v38);
  if ( (_QWORD *)v41 == v59 )
    goto LABEL_54;
  v60 = (_QWORD *)v41;
  while ( 1 )
  {
    v61 = v60;
    if ( *v60 != -8192 && *v60 != -4096 )
      break;
    v60 += 2;
    if ( v59 == v60 )
      goto LABEL_54;
  }
  if ( v60 == v59 )
  {
LABEL_54:
    v46 = 1;
  }
  else
  {
    v62 = 0;
    do
    {
      if ( v62 < *((_DWORD *)v61 + 2) )
        v62 = *((_DWORD *)v61 + 2);
      v61 += 2;
      v98 = v62;
      if ( v61 == v59 )
        break;
      while ( *v61 == -8192 || *v61 == -4096 )
      {
        v61 += 2;
        if ( v59 == v61 )
          goto LABEL_95;
      }
    }
    while ( v61 != v59 );
LABEL_95:
    v46 = v62 + 1;
  }
  v98 = v46;
  if ( v34 )
    j_j___libc_free_0((unsigned __int64)v34);
  sub_C7D6A0(0, 0, 8);
  sub_C7D6A0(0, 0, 8);
  v47 = *(_QWORD *)(v1[1] + 48LL);
  v48 = v1[2];
  v99 = &v85;
  v100 = v48;
  v101 = *(_QWORD *)(v48 + 16);
  v102 = *(_QWORD *)(v48 + 32);
  v49 = *(__int64 (**)(void))(*(_QWORD *)v101 + 128LL);
  v50 = 0;
  if ( v49 != sub_2DAC790 )
    v50 = v49();
  v103 = v50;
  v104 = v47;
  v111 = &v109;
  v112 = &v109;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v113 = 0;
  v114 = 1;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  sub_C7D6A0(0, 0, 8);
  sub_35A93B0((__int64 *)&v99);
  sub_3598EB0((__int64)&v99);
  sub_C7D6A0(v115, 24LL * v117, 8);
  sub_35E5580(v110);
  if ( v108 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v108 + 8LL))(v108);
  sub_C7D6A0(v94, 16LL * v97, 8);
  sub_C7D6A0(v90, 16LL * v92, 8);
  if ( v86 )
    j_j___libc_free_0(v86);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  sub_C7D6A0(v82, 16LL * v84, 8);
  return sub_C7D6A0(v78, 16LL * v80, 8);
}
