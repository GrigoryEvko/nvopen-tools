// Function: sub_35185B0
// Address: 0x35185b0
//
void __fastcall sub_35185B0(_QWORD *a1)
{
  __int64 v1; // rsi
  _QWORD *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 *v7; // r14
  __int64 v8; // rdi
  __int64 v9; // r13
  char v10; // al
  unsigned int v11; // eax
  __int64 v12; // r13
  unsigned __int8 v13; // al
  __int64 *v14; // r8
  __int64 *v15; // r13
  unsigned int v16; // edi
  unsigned int v17; // r9d
  __int64 *v18; // rax
  __int64 v19; // r8
  _BYTE *v20; // rax
  _BYTE *v21; // rsi
  _BYTE *v22; // rax
  __int64 v23; // r10
  unsigned __int64 v24; // rdi
  unsigned int v25; // eax
  _QWORD *v26; // rdx
  __int64 v27; // r13
  _QWORD *v28; // r11
  unsigned int v29; // ecx
  _QWORD *v30; // rbx
  unsigned __int64 v31; // rdi
  char v32; // al
  _QWORD *v33; // r8
  _QWORD *v34; // rax
  _QWORD *v35; // rbx
  __int64 v36; // rax
  _QWORD *v37; // r14
  _QWORD *v38; // r11
  __int64 v39; // r12
  __int64 v40; // r8
  __int64 v41; // r13
  __int64 v42; // r12
  _QWORD *v43; // r11
  __int64 v44; // r9
  unsigned int v45; // edi
  _QWORD *v46; // rcx
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // r15
  int v50; // esi
  int v51; // esi
  __int64 v52; // r9
  unsigned int v53; // ecx
  int v54; // edx
  _QWORD *v55; // rax
  __int64 v56; // rdi
  int v57; // edx
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // rdi
  _QWORD *v61; // r10
  unsigned int v62; // ebx
  int v63; // r9d
  __int64 v64; // rsi
  int v65; // ebx
  _QWORD *v66; // rax
  __int64 v67; // r9
  _QWORD *v68; // r13
  __int64 v69; // rdx
  int v70; // eax
  _QWORD *v71; // r10
  _QWORD *v72; // rax
  __int64 v73; // r14
  _QWORD *v74; // r13
  _QWORD *v75; // rax
  int v76; // eax
  int v77; // r10d
  __int64 v78; // rax
  int v79; // r9d
  _QWORD *v80; // r8
  _QWORD *v81; // [rsp-C8h] [rbp-C8h]
  _QWORD *v82; // [rsp-C8h] [rbp-C8h]
  __int64 v83; // [rsp-C0h] [rbp-C0h]
  int v84; // [rsp-C0h] [rbp-C0h]
  __int64 v85; // [rsp-C0h] [rbp-C0h]
  __int64 v86; // [rsp-B8h] [rbp-B8h]
  __int64 *v87; // [rsp-B0h] [rbp-B0h]
  _QWORD *v88; // [rsp-B0h] [rbp-B0h]
  __int64 v89; // [rsp-A0h] [rbp-A0h] BYREF
  unsigned __int64 v90; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v91; // [rsp-90h] [rbp-90h]
  _BYTE *v92; // [rsp-88h] [rbp-88h]
  __int64 v93; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v94; // [rsp-70h] [rbp-70h]
  __int64 v95; // [rsp-68h] [rbp-68h]
  unsigned int v96; // [rsp-60h] [rbp-60h]
  __int64 v97; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v98; // [rsp-50h] [rbp-50h]
  _BYTE *v99; // [rsp-48h] [rbp-48h]
  _BYTE *v100; // [rsp-40h] [rbp-40h]

  if ( !(_DWORD)qword_503C108 )
    return;
  v1 = 0;
  v2 = 0;
  v3 = a1[65];
  v93 = 0;
  v94 = 0;
  v4 = *(_QWORD *)(v3 + 328);
  v5 = v3 + 320;
  v95 = 0;
  v96 = 0;
  if ( v4 == v5 )
    goto LABEL_39;
  do
  {
    if ( *(_DWORD *)(v4 + 120) != 2 )
      goto LABEL_4;
    v7 = *(__int64 **)(v4 + 112);
    v8 = a1[72];
    v89 = 0;
    v9 = *v7;
    sub_2EB3EB0(v8, *v7, v4);
    if ( v10 || (v9 = v7[1], sub_2EB3EB0(a1[72], v9, v4), v32) )
      v89 = v9;
    else
      v9 = v89;
    if ( !v9 )
      goto LABEL_4;
    sub_F02DB0(&v97, 0x32u, 0x64u);
    v11 = sub_2E441D0(a1[66], v4, v89);
    if ( (unsigned int)v97 > v11 )
      goto LABEL_4;
    v12 = v89;
    v13 = sub_2FD62C0(v89);
    if ( *(_DWORD *)(v12 + 120) == 1 || !(unsigned __int8)sub_2FD64C0(a1 + 75, v13, (__int64 *)v12) )
      goto LABEL_4;
    v14 = *(__int64 **)(v89 + 64);
    v87 = &v14[*(unsigned int *)(v89 + 72)];
    if ( v14 == v87 )
    {
LABEL_17:
      if ( !v96 )
      {
        v68 = 0;
        goto LABEL_100;
      }
      v16 = v96 - 1;
      v17 = (v96 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v18 = &v94[4 * v17];
      v19 = *v18;
      if ( v4 != *v18 )
      {
        v76 = 1;
        while ( v19 != -4096 )
        {
          v77 = v76 + 1;
          v78 = v16 & (v17 + v76);
          v17 = v78;
          v18 = &v94[4 * v78];
          v19 = *v18;
          if ( *v18 == v4 )
            goto LABEL_19;
          v76 = v77;
        }
LABEL_95:
        v67 = v16 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v68 = &v94[4 * v67];
        v69 = *v68;
        if ( *v68 == v89 )
          goto LABEL_4;
        v70 = 1;
        v71 = 0;
        while ( v69 != -4096 )
        {
          if ( v69 == -8192 && !v71 )
            v71 = v68;
          v67 = v16 & ((_DWORD)v67 + v70);
          v68 = &v94[4 * v67];
          v69 = *v68;
          if ( v89 == *v68 )
            goto LABEL_4;
          ++v70;
        }
        if ( v71 )
          v68 = v71;
LABEL_100:
        v72 = sub_3514EB0((__int64)&v93, &v89, v68);
        v73 = v89;
        v72[1] = 0;
        v74 = v72;
        *v72 = v73;
        v72[2] = 0;
        v72[3] = 0;
        v75 = (_QWORD *)sub_22077B0(0x10u);
        v74[1] = v75;
        v74[3] = v75 + 2;
        *v75 = v4;
        v75[1] = v73;
        v74[2] = v75 + 2;
        goto LABEL_4;
      }
LABEL_19:
      if ( v18 == &v94[4 * v96] )
        goto LABEL_95;
      v90 = v18[1];
      v91 = (_BYTE *)v18[2];
      v92 = (_BYTE *)v18[3];
      *v18 = -8192;
      v20 = v91;
      v21 = v92;
      LODWORD(v95) = v95 - 1;
      ++HIDWORD(v95);
      v97 = v89;
      if ( v91 == v92 )
      {
        sub_2E33A40((__int64)&v90, v92, &v97);
        v21 = v92;
        v22 = v91;
      }
      else
      {
        if ( v91 )
        {
          *(_QWORD *)v91 = v89;
          v20 = v91;
          v21 = v92;
        }
        v22 = v20 + 8;
        v91 = v22;
      }
      v23 = *((_QWORD *)v22 - 1);
      v24 = v90;
      v92 = 0;
      v99 = v22;
      v91 = 0;
      v90 = 0;
      v97 = v23;
      v98 = v24;
      v100 = v21;
      if ( v96 )
      {
        v25 = (v96 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v26 = &v94[4 * v25];
        v27 = *v26;
        if ( v23 == *v26 )
        {
LABEL_26:
          if ( v24 )
            j_j___libc_free_0(v24);
          goto LABEL_28;
        }
        v79 = 1;
        v80 = 0;
        while ( v27 != -4096 )
        {
          if ( !v80 && v27 == -8192 )
            v80 = v26;
          v25 = (v96 - 1) & (v79 + v25);
          v26 = &v94[4 * v25];
          v27 = *v26;
          if ( v23 == *v26 )
            goto LABEL_26;
          ++v79;
        }
        if ( v80 )
          v26 = v80;
      }
      else
      {
        v26 = 0;
      }
      v66 = sub_3514EB0((__int64)&v93, &v97, v26);
      *v66 = v97;
      v66[1] = v98;
      v66[2] = v99;
      v66[3] = v100;
LABEL_28:
      if ( v90 )
        j_j___libc_free_0(v90);
      goto LABEL_4;
    }
    v15 = *(__int64 **)(v89 + 64);
    while ( *v15 == v4 || (unsigned __int8)sub_2FD7360(a1 + 75, v89, *v15) )
    {
      if ( v87 == ++v15 )
        goto LABEL_17;
    }
LABEL_4:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v5 != v4 );
  v28 = a1;
  v2 = v94;
  v29 = v96;
  v1 = 4LL * v96;
  if ( !(_DWORD)v95 )
    goto LABEL_31;
  v33 = &v94[v1];
  if ( v94 == &v94[v1] )
    goto LABEL_31;
  v34 = v94;
  while ( 1 )
  {
    v35 = v34;
    if ( *v34 != -8192 && *v34 != -4096 )
      break;
    v34 += 4;
    if ( v33 == v34 )
      goto LABEL_31;
  }
  if ( v34 != v33 )
  {
    v36 = (__int64)(v28 + 61);
    v37 = v28;
    v38 = &v94[v1];
    v86 = v36;
    while ( 1 )
    {
      v39 = v35[2];
      v40 = v35[1];
      if ( (unsigned int)qword_503C108 > (unsigned int)((v39 - v40) >> 3) - 1 )
        goto LABEL_50;
      v41 = *(_QWORD *)(v39 - 8);
      v42 = v39 - 8;
      v35[2] = v42;
      if ( v40 == v42 )
        goto LABEL_50;
      v88 = v38;
      v43 = v35;
      do
      {
        while ( 1 )
        {
          v48 = *((_DWORD *)v37 + 128);
          v49 = v41;
          v41 = *(_QWORD *)(v42 - 8);
          if ( !v48 )
          {
            ++v37[61];
LABEL_63:
            v81 = v43;
            v83 = v40;
            sub_35124E0(v86, 2 * v48);
            v50 = *((_DWORD *)v37 + 128);
            if ( !v50 )
              goto LABEL_127;
            v51 = v50 - 1;
            v52 = v37[62];
            v40 = v83;
            v43 = v81;
            v53 = v51 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v54 = *((_DWORD *)v37 + 126) + 1;
            v55 = (_QWORD *)(v52 + 24LL * v53);
            v56 = *v55;
            if ( v41 != *v55 )
            {
              v65 = 1;
              v61 = 0;
              while ( v56 != -4096 )
              {
                if ( v56 == -8192 && !v61 )
                  v61 = v55;
                v53 = v51 & (v65 + v53);
                v55 = (_QWORD *)(v52 + 24LL * v53);
                v56 = *v55;
                if ( v41 == *v55 )
                  goto LABEL_65;
                ++v65;
              }
LABEL_78:
              if ( v61 )
                v55 = v61;
              goto LABEL_65;
            }
            goto LABEL_65;
          }
          v44 = v37[62];
          v45 = (v48 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v46 = (_QWORD *)(v44 + 24LL * v45);
          v47 = *v46;
          if ( v41 != *v46 )
            break;
LABEL_60:
          v42 -= 8;
          if ( v40 == v42 )
            goto LABEL_68;
        }
        v84 = 1;
        v55 = 0;
        while ( v47 != -4096 )
        {
          if ( v47 != -8192 || v55 )
            v46 = v55;
          v45 = (v48 - 1) & (v84 + v45);
          v47 = *(_QWORD *)(v44 + 24LL * v45);
          if ( v41 == v47 )
            goto LABEL_60;
          ++v84;
          v55 = v46;
          v46 = (_QWORD *)(v44 + 24LL * v45);
        }
        v57 = *((_DWORD *)v37 + 126);
        if ( !v55 )
          v55 = v46;
        ++v37[61];
        v54 = v57 + 1;
        if ( 4 * v54 >= 3 * v48 )
          goto LABEL_63;
        if ( v48 - *((_DWORD *)v37 + 127) - v54 <= v48 >> 3 )
        {
          v82 = v43;
          v85 = v40;
          sub_35124E0(v86, v48);
          v58 = *((_DWORD *)v37 + 128);
          if ( !v58 )
          {
LABEL_127:
            ++*((_DWORD *)v37 + 126);
            BUG();
          }
          v59 = v58 - 1;
          v60 = v37[62];
          v61 = 0;
          v62 = v59 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v40 = v85;
          v43 = v82;
          v63 = 1;
          v54 = *((_DWORD *)v37 + 126) + 1;
          v55 = (_QWORD *)(v60 + 24LL * v62);
          v64 = *v55;
          if ( v41 != *v55 )
          {
            while ( v64 != -4096 )
            {
              if ( !v61 && v64 == -8192 )
                v61 = v55;
              v62 = v59 & (v63 + v62);
              v55 = (_QWORD *)(v60 + 24LL * v62);
              v64 = *v55;
              if ( v41 == *v55 )
                goto LABEL_65;
              ++v63;
            }
            goto LABEL_78;
          }
        }
LABEL_65:
        *((_DWORD *)v37 + 126) = v54;
        if ( *v55 != -4096 )
          --*((_DWORD *)v37 + 127);
        v42 -= 8;
        *v55 = v41;
        v55[1] = v49;
        *((_BYTE *)v55 + 16) = 1;
      }
      while ( v40 != v42 );
LABEL_68:
      v35 = v43;
      v38 = v88;
LABEL_50:
      v35 += 4;
      if ( v35 != v38 )
      {
        while ( *v35 == -8192 || *v35 == -4096 )
        {
          v35 += 4;
          if ( v38 == v35 )
            goto LABEL_54;
        }
        if ( v35 != v38 )
          continue;
      }
LABEL_54:
      v2 = v94;
      v29 = v96;
      v1 = 4LL * v96;
      break;
    }
  }
LABEL_31:
  if ( v29 )
  {
    v30 = &v2[v1];
    do
    {
      if ( *v2 != -8192 && *v2 != -4096 )
      {
        v31 = v2[1];
        if ( v31 )
          j_j___libc_free_0(v31);
      }
      v2 += 4;
    }
    while ( v30 != v2 );
    v2 = v94;
    v1 = 4LL * v96;
  }
LABEL_39:
  sub_C7D6A0((__int64)v2, v1 * 8, 8);
}
