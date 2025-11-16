// Function: sub_2BC5990
// Address: 0x2bc5990
//
__int64 __fastcall sub_2BC5990(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, __int64, __int64),
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, unsigned __int8 *, unsigned __int8 *, __int64),
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64 *, __int64, __int64),
        __int64 a8)
{
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int8 **v11; // r13
  unsigned __int8 **v12; // r14
  int v13; // ecx
  unsigned int v14; // edx
  unsigned __int8 *v15; // rsi
  unsigned __int8 v16; // al
  int v17; // edx
  __int64 v18; // rdi
  unsigned __int8 **v19; // r15
  unsigned __int8 **v20; // r12
  int v21; // eax
  __int64 v22; // rcx
  int v23; // eax
  unsigned int v24; // edx
  unsigned __int8 *v25; // r14
  unsigned __int8 **v26; // r13
  unsigned int v27; // edx
  unsigned __int8 *v28; // rsi
  unsigned __int8 **v29; // r12
  unsigned int v30; // r15d
  unsigned __int8 v31; // al
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // rax
  unsigned __int8 v39; // al
  unsigned __int8 **v40; // r15
  unsigned __int8 **v41; // r14
  int v42; // ecx
  int v43; // r10d
  unsigned int v44; // edx
  unsigned __int8 *v45; // rdi
  unsigned __int8 *v46; // r12
  unsigned __int8 v47; // al
  int v48; // ecx
  __int64 v49; // rsi
  unsigned __int8 **v50; // r13
  __int64 v51; // r15
  unsigned __int8 **v52; // rbx
  int v53; // eax
  __int64 v54; // rsi
  int v55; // eax
  unsigned int v56; // edx
  unsigned __int8 *v57; // rdi
  unsigned __int8 **v58; // r14
  unsigned int v59; // edx
  unsigned __int8 *v60; // rcx
  unsigned __int8 **v61; // r10
  __int64 v62; // rdx
  char v63; // al
  char v64; // cl
  int v65; // edi
  int v66; // eax
  int v67; // edi
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  int v70; // r10d
  int v71; // eax
  int v72; // edi
  __int64 v73; // rax
  unsigned __int64 v74; // rdx
  __int64 *v76; // r15
  unsigned __int8 ***v77; // r11
  int v78; // ecx
  unsigned int v79; // edx
  __int64 v80; // rdi
  int v81; // ecx
  __int64 v82; // rsi
  __int64 *v83; // r15
  __int64 *v84; // r11
  unsigned __int8 v85; // r10
  int v86; // ecx
  unsigned int v87; // edx
  __int64 v88; // rdi
  int v89; // ecx
  __int64 v90; // rsi
  int v91; // r10d
  unsigned __int8 **v92; // [rsp+0h] [rbp-150h]
  unsigned __int8 **v93; // [rsp+8h] [rbp-148h]
  __int64 v94; // [rsp+8h] [rbp-148h]
  __int64 *v95; // [rsp+8h] [rbp-148h]
  __int64 v96; // [rsp+20h] [rbp-130h]
  unsigned __int8 v97; // [rsp+20h] [rbp-130h]
  unsigned __int8 v98; // [rsp+30h] [rbp-120h]
  unsigned __int8 **v99; // [rsp+48h] [rbp-108h]
  unsigned __int8 **v100; // [rsp+48h] [rbp-108h]
  unsigned __int8 v101; // [rsp+48h] [rbp-108h]
  unsigned __int8 ***v102; // [rsp+48h] [rbp-108h]
  __int64 v103; // [rsp+48h] [rbp-108h]
  unsigned __int8 **v106; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v107; // [rsp+68h] [rbp-E8h]
  _BYTE v108[48]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v109; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-A8h]
  _BYTE v111[48]; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 *v112; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v113; // [rsp+E8h] [rbp-68h]
  _BYTE v114[96]; // [rsp+F0h] [rbp-60h] BYREF

  sub_2BC58E0(a1, a2, a3);
  v11 = *(unsigned __int8 ***)a1;
  v106 = (unsigned __int8 **)v108;
  v107 = 0x600000000LL;
  v110 = 0x600000000LL;
  v12 = &v11[*(unsigned int *)(a1 + 8)];
  v109 = (__int64 *)v111;
  v98 = 0;
  if ( v11 == v12 )
    return v98;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *v11;
        v16 = **v11;
        if ( v16 > 0x1Cu )
        {
          v17 = *(_DWORD *)(a6 + 2000);
          v18 = *(_QWORD *)(a6 + 1984);
          if ( !v17 )
            goto LABEL_8;
          v13 = v17 - 1;
          v10 = 1;
          v14 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v9 = *(_QWORD *)(v18 + 8LL * v14);
          if ( v15 != (unsigned __int8 *)v9 )
            break;
        }
LABEL_4:
        ++v11;
LABEL_5:
        LODWORD(v110) = 0;
        if ( v12 == v11 )
          goto LABEL_77;
      }
      while ( v9 != -4096 )
      {
        v14 = v13 & (v10 + v14);
        v9 = *(_QWORD *)(v18 + 8LL * v14);
        if ( v15 == (unsigned __int8 *)v9 )
          goto LABEL_4;
        v10 = (unsigned int)(v10 + 1);
      }
LABEL_8:
      if ( v12 != v11 )
      {
        v99 = v11;
        v19 = v11 + 1;
        v20 = v12;
        while ( 1 )
        {
          v26 = v19 - 1;
          if ( v16 > 0x1Cu )
          {
            v21 = *(_DWORD *)(a6 + 2000);
            v22 = *(_QWORD *)(a6 + 1984);
            if ( v21 )
            {
              v23 = v21 - 1;
              v24 = v23 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
              v25 = *(unsigned __int8 **)(v22 + 8LL * v24);
              if ( v15 == v25 )
              {
LABEL_12:
                v26 = v19;
                goto LABEL_13;
              }
              v65 = 1;
              while ( v25 != (unsigned __int8 *)-4096LL )
              {
                v9 = (unsigned int)(v65 + 1);
                v24 = v23 & (v65 + v24);
                v25 = *(unsigned __int8 **)(v22 + 8LL * v24);
                if ( v15 == v25 )
                  goto LABEL_12;
                ++v65;
              }
            }
            if ( !a4(a5, v15, *v99, v22) )
            {
LABEL_18:
              v12 = v20;
              v29 = v26;
              v11 = v99;
              goto LABEL_19;
            }
            v25 = *(v19 - 1);
            if ( *v25 > 0x1Cu )
              break;
          }
          v26 = v19;
          if ( v20 == v19 )
            goto LABEL_18;
LABEL_15:
          v15 = *v19++;
          v16 = *v15;
        }
        v66 = *(_DWORD *)(a6 + 2000);
        v22 = *(_QWORD *)(a6 + 1984);
        v26 = v19;
        if ( v66 )
        {
          v23 = v66 - 1;
LABEL_13:
          v27 = v23 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v28 = *(unsigned __int8 **)(v22 + 8LL * v27);
          if ( v25 == v28 )
          {
LABEL_14:
            if ( v20 == v19 )
              goto LABEL_18;
            goto LABEL_15;
          }
          v67 = 1;
          while ( v28 != (unsigned __int8 *)-4096LL )
          {
            v9 = (unsigned int)(v67 + 1);
            v27 = v23 & (v67 + v27);
            v28 = *(unsigned __int8 **)(v22 + 8LL * v27);
            if ( v25 == v28 )
              goto LABEL_14;
            ++v67;
          }
        }
        v68 = (unsigned int)v110;
        v69 = (unsigned int)v110 + 1LL;
        if ( v69 > HIDWORD(v110) )
        {
          sub_C8D5F0((__int64)&v109, v111, v69, 8u, v9, v10);
          v68 = (unsigned int)v110;
        }
        v109[v68] = (__int64)v25;
        LODWORD(v110) = v110 + 1;
        goto LABEL_14;
      }
      v29 = v12;
LABEL_19:
      v30 = v110;
      if ( (unsigned int)v110 <= 1 )
      {
        sub_2B49BC0(a6, *v11);
        goto LABEL_83;
      }
      v31 = a7(a8, v109, (unsigned int)v110, 1);
      if ( v31 )
      {
        v101 = v31;
        sub_2B38BA0((__int64)&v109, (__int64)&v106, v32, v33, v34, v35);
        v83 = v109;
        LODWORD(v107) = 0;
        v10 = v101;
        v84 = &v109[(unsigned int)v110];
        if ( v109 == v84 )
        {
          v98 = v101;
          v11 = v29;
          goto LABEL_5;
        }
        v38 = 0;
        v85 = v101;
        while ( 2 )
        {
          while ( 1 )
          {
            v9 = *v83;
            if ( *(_BYTE *)*v83 <= 0x1Cu )
              break;
            v89 = *(_DWORD *)(a6 + 2000);
            v90 = *(_QWORD *)(a6 + 1984);
            if ( v89 )
            {
              v86 = v89 - 1;
              v87 = v86 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v88 = *(_QWORD *)(v90 + 8LL * v87);
              if ( v9 == v88 )
                break;
              v10 = 1;
              while ( v88 != -4096 )
              {
                v87 = v86 & (v10 + v87);
                v88 = *(_QWORD *)(v90 + 8LL * v87);
                if ( v9 == v88 )
                  goto LABEL_111;
                v10 = (unsigned int)(v10 + 1);
              }
            }
            if ( v38 + 1 > (unsigned __int64)HIDWORD(v107) )
            {
              v95 = v84;
              v97 = v85;
              v103 = *v83;
              sub_C8D5F0((__int64)&v106, v108, v38 + 1, 8u, v9, v10);
              v38 = (unsigned int)v107;
              v84 = v95;
              v85 = v97;
              v9 = v103;
            }
            ++v83;
            v106[v38] = (unsigned __int8 *)v9;
            v38 = (unsigned int)(v107 + 1);
            LODWORD(v107) = v107 + 1;
            if ( v84 == v83 )
            {
LABEL_117:
              v98 = v85;
              goto LABEL_25;
            }
          }
LABEL_111:
          if ( v84 == ++v83 )
            goto LABEL_117;
          continue;
        }
      }
      v36 = sub_2B49BC0(a6, *v11);
      v9 = v36;
      v37 = *(_DWORD *)(a6 + 3360) / v36;
      if ( v37 < 2 )
        v37 = 2;
      if ( v30 >= v37 )
      {
        LODWORD(v38) = v107;
        goto LABEL_25;
      }
LABEL_83:
      v38 = (unsigned int)v107;
      if ( !(_DWORD)v107 || *((_QWORD *)*v106 + 1) == *((_QWORD *)*v11 + 1) )
      {
        v76 = v109;
        v10 = (__int64)&v109[(unsigned int)v110];
        if ( v109 != (__int64 *)v10 )
        {
          v77 = &v106;
          while ( 1 )
          {
            v9 = *v76;
            if ( *(_BYTE *)*v76 > 0x1Cu )
            {
              v81 = *(_DWORD *)(a6 + 2000);
              v82 = *(_QWORD *)(a6 + 1984);
              if ( !v81 )
                goto LABEL_91;
              v78 = v81 - 1;
              v79 = v78 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v80 = *(_QWORD *)(v82 + 8LL * v79);
              if ( v9 != v80 )
                break;
            }
LABEL_88:
            if ( (__int64 *)v10 == ++v76 )
              goto LABEL_25;
          }
          v91 = 1;
          while ( v80 != -4096 )
          {
            v79 = v78 & (v91 + v79);
            v80 = *(_QWORD *)(v82 + 8LL * v79);
            if ( v9 == v80 )
              goto LABEL_88;
            ++v91;
          }
LABEL_91:
          if ( v38 + 1 > (unsigned __int64)HIDWORD(v107) )
          {
            v94 = v10;
            v96 = *v76;
            v102 = v77;
            sub_C8D5F0((__int64)v77, v108, v38 + 1, 8u, v9, v10);
            v38 = (unsigned int)v107;
            v10 = v94;
            v9 = v96;
            v77 = v102;
          }
          v106[v38] = (unsigned __int8 *)v9;
          v38 = (unsigned int)(v107 + 1);
          LODWORD(v107) = v107 + 1;
          goto LABEL_88;
        }
      }
LABEL_25:
      if ( (unsigned int)v38 <= 1 )
      {
        v11 = v29;
        goto LABEL_5;
      }
      if ( v12 == v29 || *((_QWORD *)*v29 + 1) != *((_QWORD *)*v11 + 1) )
        break;
      LODWORD(v110) = 0;
      v11 = v29;
    }
    v39 = a7(a8, (__int64 *)v106, (unsigned int)v38, 0);
    if ( v39 )
    {
      v98 = v39;
LABEL_75:
      LODWORD(v107) = 0;
      v11 = v29;
      goto LABEL_76;
    }
    v10 = (__int64)v106;
    v112 = (__int64 *)v114;
    v113 = 0x600000000LL;
    v9 = (__int64)&v106[(unsigned int)v107];
    if ( v106 == (unsigned __int8 **)v9 )
      goto LABEL_75;
    v93 = v12;
    v40 = v106;
    v41 = &v106[(unsigned int)v107];
    v92 = v29;
    do
    {
      v46 = *v40;
      v47 = **v40;
      if ( v47 <= 0x1Cu )
      {
LABEL_32:
        ++v40;
        goto LABEL_33;
      }
      v48 = *(_DWORD *)(a6 + 2000);
      v49 = *(_QWORD *)(a6 + 1984);
      if ( v48 )
      {
        v42 = v48 - 1;
        v43 = 1;
        v44 = v42 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v45 = *(unsigned __int8 **)(v49 + 8LL * v44);
        if ( v46 == v45 )
          goto LABEL_32;
        while ( v45 != (unsigned __int8 *)-4096LL )
        {
          v9 = (unsigned int)(v43 + 1);
          v44 = v42 & (v43 + v44);
          v45 = *(unsigned __int8 **)(v49 + 8LL * v44);
          if ( v46 == v45 )
            goto LABEL_32;
          ++v43;
        }
      }
      if ( v41 != v40 )
      {
        v100 = v40;
        v50 = v40 + 1;
        v51 = a6;
        v52 = v41;
        while ( 1 )
        {
          if ( v47 > 0x1Cu )
          {
            v53 = *(_DWORD *)(v51 + 2000);
            v54 = *(_QWORD *)(v51 + 1984);
            if ( v53 )
            {
              v55 = v53 - 1;
              v56 = v55 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
              v57 = *(unsigned __int8 **)(v54 + 8LL * v56);
              if ( v57 == v46 )
              {
LABEL_40:
                v58 = v50;
                goto LABEL_41;
              }
              v70 = 1;
              while ( v57 != (unsigned __int8 *)-4096LL )
              {
                v9 = (unsigned int)(v70 + 1);
                v56 = v55 & (v70 + v56);
                v57 = *(unsigned __int8 **)(v54 + 8LL * v56);
                if ( v57 == v46 )
                  goto LABEL_40;
                ++v70;
              }
            }
            if ( !((unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, unsigned __int8 *))a4)(a5, v46, *v100) )
            {
              v62 = (unsigned int)v113;
              v41 = v52;
              a6 = v51;
              v40 = v50 - 1;
              if ( (unsigned int)v113 <= 1 )
                goto LABEL_33;
              goto LABEL_49;
            }
            v46 = *(v50 - 1);
            if ( *v46 > 0x1Cu )
              break;
          }
          v58 = v50;
          if ( v52 == v50 )
          {
LABEL_46:
            v61 = v58;
            v41 = v52;
            a6 = v51;
            goto LABEL_47;
          }
LABEL_43:
          v46 = *v50++;
          v47 = *v46;
        }
        v71 = *(_DWORD *)(v51 + 2000);
        v54 = *(_QWORD *)(v51 + 1984);
        v58 = v50;
        if ( v71 )
        {
          v55 = v71 - 1;
LABEL_41:
          v59 = v55 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v60 = *(unsigned __int8 **)(v54 + 8LL * v59);
          if ( v46 == v60 )
          {
LABEL_42:
            if ( v52 == v50 )
              goto LABEL_46;
            goto LABEL_43;
          }
          v72 = 1;
          while ( v60 != (unsigned __int8 *)-4096LL )
          {
            v9 = (unsigned int)(v72 + 1);
            v59 = v55 & (v72 + v59);
            v60 = *(unsigned __int8 **)(v54 + 8LL * v59);
            if ( v46 == v60 )
              goto LABEL_42;
            ++v72;
          }
        }
        v73 = (unsigned int)v113;
        v74 = (unsigned int)v113 + 1LL;
        if ( v74 > HIDWORD(v113) )
        {
          sub_C8D5F0((__int64)&v112, v114, v74, 8u, v9, v10);
          v73 = (unsigned int)v113;
        }
        v112[v73] = (__int64)v46;
        LODWORD(v113) = v113 + 1;
        goto LABEL_42;
      }
      v61 = v41;
LABEL_47:
      v62 = (unsigned int)v113;
      if ( (unsigned int)v113 <= 1 )
        break;
      v40 = v61;
LABEL_49:
      v63 = a7(a8, v112, v62, 0);
      v64 = v98;
      if ( v63 )
        v64 = v63;
      v98 = v64;
LABEL_33:
      LODWORD(v113) = 0;
    }
    while ( v41 != v40 );
    v12 = v93;
    v29 = v92;
    if ( v112 == (__int64 *)v114 )
      goto LABEL_75;
    _libc_free((unsigned __int64)v112);
    v11 = v92;
    LODWORD(v107) = 0;
LABEL_76:
    LODWORD(v110) = 0;
  }
  while ( v12 != v11 );
LABEL_77:
  if ( v109 != (__int64 *)v111 )
    _libc_free((unsigned __int64)v109);
  if ( v106 != (unsigned __int8 **)v108 )
    _libc_free((unsigned __int64)v106);
  return v98;
}
