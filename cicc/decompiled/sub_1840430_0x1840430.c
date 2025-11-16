// Function: sub_1840430
// Address: 0x1840430
//
__int64 __fastcall sub_1840430(__int64 *a1)
{
  size_t *v1; // rdi
  __int64 *v2; // r15
  __int64 *v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rsi
  unsigned int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  _QWORD *v18; // rax
  _BYTE *v19; // r13
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rsi
  __int64 *v25; // rax
  __int64 *v26; // r12
  __int64 v27; // r15
  __int64 *v28; // rbx
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // rdi
  _BYTE *v33; // rdx
  unsigned __int8 v34; // dl
  unsigned __int64 v35; // rax
  unsigned __int64 *v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  unsigned int v39; // eax
  size_t v40; // rdx
  __int64 v41; // r9
  _BYTE *v42; // rsi
  _BYTE *v43; // rax
  unsigned __int64 v44; // r14
  size_t *v45; // r13
  size_t v46; // rdx
  size_t *v47; // rdx
  __int64 v48; // rax
  __int64 *v49; // rax
  int v50; // edx
  _BYTE *v51; // rax
  unsigned __int64 v52; // rdx
  int v53; // r10d
  size_t v54; // [rsp+8h] [rbp-748h]
  size_t v55; // [rsp+8h] [rbp-748h]
  size_t v56; // [rsp+8h] [rbp-748h]
  int v57; // [rsp+24h] [rbp-72Ch]
  unsigned __int8 v59; // [rsp+28h] [rbp-728h]
  char *v60[2]; // [rsp+38h] [rbp-718h] BYREF
  __int64 v61; // [rsp+48h] [rbp-708h]
  __int64 v62; // [rsp+50h] [rbp-700h] BYREF
  char *v63[2]; // [rsp+58h] [rbp-6F8h] BYREF
  __int64 v64; // [rsp+68h] [rbp-6E8h]
  __int64 (__fastcall **v65)(); // [rsp+70h] [rbp-6E0h] BYREF
  int v66; // [rsp+78h] [rbp-6D8h]
  _QWORD v67[2]; // [rsp+80h] [rbp-6D0h] BYREF
  __int64 v68; // [rsp+90h] [rbp-6C0h]
  int v69; // [rsp+98h] [rbp-6B8h]
  _QWORD v70[2]; // [rsp+A0h] [rbp-6B0h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-6A0h]
  int v72; // [rsp+B8h] [rbp-698h]
  _QWORD v73[2]; // [rsp+C0h] [rbp-690h] BYREF
  __int64 v74; // [rsp+D0h] [rbp-680h]
  __int64 v75; // [rsp+D8h] [rbp-678h]
  __int64 *v76; // [rsp+E0h] [rbp-670h]
  __int64 *v77; // [rsp+E8h] [rbp-668h]
  __int64 v78; // [rsp+F0h] [rbp-660h]
  int v79; // [rsp+F8h] [rbp-658h]
  _BYTE v80[256]; // [rsp+100h] [rbp-650h] BYREF
  size_t v81; // [rsp+200h] [rbp-550h] BYREF
  size_t *v82; // [rsp+208h] [rbp-548h] BYREF
  _QWORD *v83; // [rsp+210h] [rbp-540h]
  __int64 v84; // [rsp+218h] [rbp-538h]
  unsigned int v85; // [rsp+220h] [rbp-530h]
  __int64 v86; // [rsp+228h] [rbp-528h] BYREF
  _BYTE *v87; // [rsp+230h] [rbp-520h]
  _BYTE *v88; // [rsp+238h] [rbp-518h]
  __int64 v89; // [rsp+240h] [rbp-510h]
  int v90; // [rsp+248h] [rbp-508h]
  _BYTE v91[128]; // [rsp+250h] [rbp-500h] BYREF
  _BYTE *v92; // [rsp+2D0h] [rbp-480h]
  __int64 v93; // [rsp+2D8h] [rbp-478h]
  _BYTE v94[512]; // [rsp+2E0h] [rbp-470h] BYREF
  _BYTE *v95; // [rsp+4E0h] [rbp-270h]
  __int64 v96; // [rsp+4E8h] [rbp-268h]
  _BYTE v97[520]; // [rsp+4F0h] [rbp-260h] BYREF
  int v98; // [rsp+6F8h] [rbp-58h] BYREF
  __int64 v99; // [rsp+700h] [rbp-50h]
  int *v100; // [rsp+708h] [rbp-48h]
  int *v101; // [rsp+710h] [rbp-40h]
  __int64 v102; // [rsp+718h] [rbp-38h]

  v65 = (__int64 (__fastcall **)())&unk_49F0D08;
  v81 = 3;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v62 = 2;
  v63[0] = 0;
  v63[1] = 0;
  v64 = 0;
  v60[0] = 0;
  v60[1] = 0;
  v61 = 0;
  v66 = 0;
  v67[0] = 0;
  v67[1] = 0;
  v68 = 0;
  v69 = 0;
  v70[0] = 0;
  v70[1] = 0;
  v71 = 0;
  v72 = 0;
  v73[0] = 0;
  v73[1] = 0;
  v74 = 0;
  sub_183B0B0((__int64)v67, v60);
  v69 = v62;
  sub_183B0B0((__int64)v70, v63);
  v72 = v81;
  sub_183B0B0((__int64)v73, (char **)&v82);
  if ( v60[0] )
    j_j___libc_free_0(v60[0], v61 - (unsigned __int64)v60[0]);
  if ( v63[0] )
    j_j___libc_free_0(v63[0], v64 - (unsigned __int64)v63[0]);
  v1 = v82;
  if ( v82 )
    j_j___libc_free_0(v82, v84 - (_QWORD)v82);
  v75 = 0;
  v65 = off_4985248;
  v76 = (__int64 *)v80;
  v77 = (__int64 *)v80;
  v87 = v91;
  v88 = v91;
  v92 = v94;
  v93 = 0x4000000000LL;
  v96 = 0x4000000000LL;
  v100 = &v98;
  v101 = &v98;
  v78 = 32;
  v2 = (__int64 *)a1[4];
  v3 = a1 + 3;
  v79 = 0;
  v81 = (size_t)&v65;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v89 = 16;
  v90 = 0;
  v95 = v97;
  v98 = 0;
  v99 = 0;
  v102 = 0;
  if ( v2 != a1 + 3 )
  {
    do
    {
      while ( 1 )
      {
        v4 = (__int64)(v2 - 7);
        if ( !v2 )
          v4 = 0;
        v1 = (size_t *)v4;
        if ( !sub_15E4F60(v4) )
        {
          v1 = (size_t *)v4;
          if ( !(unsigned __int8)sub_387DFE0(v4) )
            break;
        }
        v2 = (__int64 *)v2[1];
        if ( v3 == v2 )
          goto LABEL_17;
      }
      v9 = *(_QWORD *)(v4 + 80);
      v1 = &v81;
      if ( v9 )
        v9 -= 24;
      sub_183B530((__int64)&v81, v9, v5, v6, v7, v8);
      v2 = (__int64 *)v2[1];
    }
    while ( v3 != v2 );
LABEL_17:
    v10 = v96;
    v11 = v93;
    if ( !(_DWORD)v96 )
      goto LABEL_43;
    if ( !(_DWORD)v93 )
      goto LABEL_34;
    while ( 1 )
    {
      v12 = v11--;
      v13 = *(_QWORD *)&v92[8 * v12 - 8];
      LODWORD(v93) = v11;
      v14 = *(_QWORD *)(v13 + 8);
      if ( v14 )
        break;
LABEL_32:
      if ( !v11 )
      {
        v10 = v96;
        if ( !(_DWORD)v96 )
          goto LABEL_44;
        do
        {
LABEL_34:
          v20 = v10--;
          v21 = *(_QWORD *)&v95[8 * v20 - 8];
          LODWORD(v96) = v10;
          v22 = *(_QWORD *)(v21 + 48);
          v23 = v21 + 40;
          if ( v22 != v23 )
          {
            do
            {
              while ( 1 )
              {
                if ( !v22 )
                  BUG();
                v24 = v22 - 24;
                v1 = &v81;
                if ( *(_BYTE *)(v22 - 8) != 77 )
                  break;
                sub_183D9D0(&v81, v24);
                v22 = *(_QWORD *)(v22 + 8);
                if ( v23 == v22 )
                  goto LABEL_40;
              }
              sub_183FEC0(&v81, v24);
              v22 = *(_QWORD *)(v22 + 8);
            }
            while ( v23 != v22 );
LABEL_40:
            v10 = v96;
          }
        }
        while ( v10 );
        v11 = v93;
LABEL_43:
        if ( !v11 )
          goto LABEL_44;
      }
    }
    while ( 1 )
    {
      v1 = (size_t *)v14;
      v15 = sub_1648700(v14);
      v16 = (__int64)v15;
      if ( *((_BYTE *)v15 + 16) <= 0x17u )
        goto LABEL_22;
      v17 = v15[5];
      v18 = v87;
      if ( v88 == v87 )
        break;
      v1 = (size_t *)&v86;
      v19 = &v88[8 * (unsigned int)v89];
      v18 = sub_16CC9F0((__int64)&v86, v17);
      if ( v17 == *v18 )
      {
        if ( v88 == v87 )
          v33 = &v88[8 * HIDWORD(v89)];
        else
          v33 = &v88[8 * (unsigned int)v89];
        goto LABEL_77;
      }
      if ( v88 == v87 )
      {
        v18 = &v88[8 * HIDWORD(v89)];
        v33 = v18;
        goto LABEL_77;
      }
      v18 = &v88[8 * (unsigned int)v89];
LABEL_28:
      if ( v18 == (_QWORD *)v19 )
        goto LABEL_22;
      v1 = &v81;
      if ( *(_BYTE *)(v16 + 16) == 77 )
      {
        sub_183D9D0(&v81, v16);
LABEL_22:
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
          goto LABEL_31;
      }
      else
      {
        sub_183FEC0(&v81, v16);
        v14 = *(_QWORD *)(v14 + 8);
        if ( !v14 )
        {
LABEL_31:
          v11 = v93;
          goto LABEL_32;
        }
      }
    }
    v19 = &v87[8 * HIDWORD(v89)];
    if ( v87 == v19 )
    {
      v33 = v87;
    }
    else
    {
      do
      {
        if ( v17 == *v18 )
          break;
        ++v18;
      }
      while ( v19 != (_BYTE *)v18 );
      v33 = &v87[8 * HIDWORD(v89)];
    }
LABEL_77:
    while ( v33 != (_BYTE *)v18 )
    {
      if ( *v18 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v18;
    }
    goto LABEL_28;
  }
LABEL_44:
  v62 = *a1;
  v25 = v77;
  if ( v77 == v76 )
    v26 = &v77[HIDWORD(v78)];
  else
    v26 = &v77[(unsigned int)v78];
  if ( v77 != v26 )
  {
    while ( 1 )
    {
      v27 = *v25;
      v28 = v25;
      if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v26 == ++v25 )
        goto LABEL_49;
    }
    if ( v26 != v25 )
    {
      v34 = *(_BYTE *)(v27 + 16);
      v59 = 0;
      if ( v34 <= 0x17u )
        goto LABEL_108;
      do
      {
        if ( v34 == 78 )
        {
          v52 = v27 | 4;
        }
        else
        {
          v35 = 0;
          if ( v34 != 29 )
            goto LABEL_88;
          v52 = v27 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v35 = v52 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v52 & 4) == 0 )
          goto LABEL_88;
        v36 = (unsigned __int64 *)(v35 - 24);
LABEL_89:
        v37 = v85;
        v38 = *v36;
        if ( v85 )
        {
          v1 = (size_t *)(v38 & 0xFFFFFFFFFFFFFFF9LL);
          v39 = (v85 - 1) & (v38 & 0xFFFFFFF9 ^ (v38 >> 9));
          v40 = (size_t)&v83[5 * v39];
          v41 = *(_QWORD *)v40;
          if ( v1 == *(size_t **)v40 )
          {
LABEL_91:
            if ( (_QWORD *)v40 != &v83[5 * v85] )
            {
              v42 = *(_BYTE **)(v40 + 16);
              v57 = *(_DWORD *)(v40 + 8);
              v43 = *(_BYTE **)(v40 + 24);
              v44 = v43 - v42;
              if ( v43 == v42 )
              {
                v46 = 0;
                v45 = 0;
              }
              else
              {
                if ( v44 > 0x7FFFFFFFFFFFFFF8LL )
                  goto LABEL_127;
                v1 = (size_t *)(*(_QWORD *)(v40 + 24) - (_QWORD)v42);
                v54 = v40;
                v45 = (size_t *)sub_22077B0(v44);
                v43 = *(_BYTE **)(v54 + 24);
                v42 = *(_BYTE **)(v54 + 16);
                v46 = v43 - v42;
              }
              if ( v42 != v43 )
                goto LABEL_96;
              goto LABEL_97;
            }
          }
          else
          {
            v50 = 1;
            while ( v41 != -2 )
            {
              v53 = v50 + 1;
              v39 = (v85 - 1) & (v50 + v39);
              v40 = (size_t)&v83[5 * v39];
              v41 = *(_QWORD *)v40;
              if ( v1 == *(size_t **)v40 )
                goto LABEL_91;
              v50 = v53;
            }
          }
        }
        v40 = v81;
        v42 = *(_BYTE **)(v81 + 80);
        v56 = v81;
        v57 = *(_DWORD *)(v81 + 72);
        v51 = *(_BYTE **)(v81 + 88);
        v44 = v51 - v42;
        if ( v51 == v42 )
        {
          v46 = 0;
          v45 = 0;
        }
        else
        {
          if ( v44 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_127:
            sub_4261EA(v1, v42, v40);
          v1 = (size_t *)(*(_QWORD *)(v81 + 88) - (_QWORD)v42);
          v45 = (size_t *)sub_22077B0(v44);
          v51 = *(_BYTE **)(v56 + 88);
          v42 = *(_BYTE **)(v56 + 80);
          v46 = v51 - v42;
        }
        if ( v51 != v42 )
        {
LABEL_96:
          v1 = v45;
          v55 = v46;
          memmove(v45, v42, v46);
          v46 = v55;
        }
LABEL_97:
        v47 = (size_t *)((char *)v45 + v46);
        if ( v57 != 1 || v47 == v45 )
        {
          if ( v45 )
          {
            v1 = v45;
            j_j___libc_free_0(v45, v44);
          }
        }
        else
        {
          v48 = sub_161C3B0(&v62, (__int64 *)v45, v47 - v45, v37);
          v1 = (size_t *)v27;
          sub_1625C10(v27, 23, v48);
          if ( v45 )
          {
            v1 = v45;
            j_j___libc_free_0(v45, v44);
          }
          v59 = 1;
        }
        v49 = v28 + 1;
        if ( v28 + 1 == v26 )
          goto LABEL_50;
        while ( 1 )
        {
          v27 = *v49;
          v28 = v49;
          if ( (unsigned __int64)*v49 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v26 == ++v49 )
            goto LABEL_50;
        }
        if ( v26 == v49 )
          goto LABEL_50;
        v34 = *(_BYTE *)(v27 + 16);
      }
      while ( v34 > 0x17u );
LABEL_108:
      v35 = 0;
LABEL_88:
      v36 = (unsigned __int64 *)(v35 - 72);
      goto LABEL_89;
    }
  }
LABEL_49:
  v59 = 0;
LABEL_50:
  sub_183B210(v99);
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  if ( v88 != v87 )
    _libc_free((unsigned __int64)v88);
  if ( v85 )
  {
    v29 = v83;
    v30 = &v83[5 * v85];
    do
    {
      if ( *v29 != -16 && *v29 != -2 )
      {
        v31 = v29[2];
        if ( v31 )
          j_j___libc_free_0(v31, v29[4] - v31);
      }
      v29 += 5;
    }
    while ( v30 != v29 );
  }
  j___libc_free_0(v83);
  if ( v77 != v76 )
    _libc_free((unsigned __int64)v77);
  v65 = (__int64 (__fastcall **)())&unk_49F0D08;
  if ( v73[0] )
    j_j___libc_free_0(v73[0], v74 - v73[0]);
  if ( v70[0] )
    j_j___libc_free_0(v70[0], v71 - v70[0]);
  if ( v67[0] )
    j_j___libc_free_0(v67[0], v68 - v67[0]);
  return v59;
}
