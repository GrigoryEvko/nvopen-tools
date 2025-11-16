// Function: sub_16CE370
// Address: 0x16ce370
//
__int64 (__fastcall *__fastcall sub_16CE370(
        __int64 a1,
        const char *a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5))(__int64 a1)
{
  const char *v5; // r9
  char v6; // r15
  char v7; // r14
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 (__fastcall *v11)(__int64); // rdx
  __int64 (*v12)(); // rax
  char v13; // bl
  unsigned int v14; // eax
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 (__fastcall *result)(__int64); // rax
  unsigned __int8 *v20; // rsi
  char *v21; // rcx
  __int64 v22; // rdx
  char *v23; // rax
  __int64 (__fastcall *v24)(__int64); // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64); // rdx
  __int64 (__fastcall *v28)(__int64); // rax
  size_t v29; // rax
  _WORD *v30; // rdi
  size_t v31; // r15
  unsigned __int64 v32; // rax
  _QWORD *v33; // r8
  _BYTE *v34; // rdx
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // r15
  unsigned int *v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rdi
  size_t v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rbx
  unsigned __int64 *v52; // r12
  unsigned __int64 v53; // r14
  __int64 *v54; // r13
  int v55; // esi
  unsigned __int64 v56; // rcx
  __int64 v57; // r9
  unsigned __int64 v58; // rax
  size_t v59; // rdx
  unsigned int v60; // edx
  size_t v61; // rdx
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rsi
  unsigned __int64 v66; // r15
  int v67; // r14d
  __int64 v68; // rcx
  __int64 v69; // rbx
  _BYTE *v70; // rax
  _BYTE *v71; // rax
  char *v72; // rdx
  unsigned __int64 v73; // r14
  __int64 v74; // r15
  unsigned __int64 v75; // rbx
  _QWORD *v76; // rax
  __int64 v77; // rsi
  _BYTE *v78; // rax
  char v79; // al
  char v80; // bl
  __int64 (__fastcall *v81)(__int64); // rax
  __int64 (__fastcall *v82)(__int64); // rax
  __int64 v83; // rdx
  __int64 v84; // rdx
  char v85; // al
  __int64 v86; // rsi
  _BYTE *v87; // rax
  __int64 (__fastcall *v88)(__int64); // rax
  __int64 (__fastcall *v89)(__int64); // rax
  _QWORD *v90; // rdx
  __int64 v91; // rax
  unsigned __int64 v92; // r10
  __int64 v93; // rax
  unsigned __int64 v94; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v95; // [rsp+8h] [rbp-D8h]
  __int64 v96; // [rsp+8h] [rbp-D8h]
  __int64 v97; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v98; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v99; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v100; // [rsp+18h] [rbp-C8h]
  _QWORD *v101; // [rsp+20h] [rbp-C0h]
  __int64 v102; // [rsp+28h] [rbp-B8h]
  char v103; // [rsp+33h] [rbp-ADh]
  unsigned int v104; // [rsp+34h] [rbp-ACh]
  unsigned __int64 v105; // [rsp+50h] [rbp-90h]
  __int64 src; // [rsp+58h] [rbp-88h]
  const char *srca; // [rsp+58h] [rbp-88h]
  char srcb; // [rsp+58h] [rbp-88h]
  const char *srcc; // [rsp+58h] [rbp-88h]
  _QWORD v110[2]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v111; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v112; // [rsp+78h] [rbp-68h]
  _QWORD v113[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v114; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 v115; // [rsp+98h] [rbp-48h]
  _QWORD v116[8]; // [rsp+A0h] [rbp-40h] BYREF

  v5 = a2;
  v6 = a4;
  v7 = a5;
  v9 = a3;
  v10 = *a3;
  v11 = *(__int64 (__fastcall **)(__int64))(*a3 + 48LL);
  if ( v11 == sub_1548960 )
  {
    v12 = *(__int64 (**)())(v10 + 40);
    if ( v12 == sub_16CD2B0 )
    {
LABEL_3:
      v13 = 0;
      goto LABEL_4;
    }
    v85 = ((__int64 (__fastcall *)(_QWORD *, const char *, __int64 (__fastcall *)(__int64), __int64, __int64, const char *))v12)(
            v9,
            a2,
            v11,
            a4,
            a5,
            a2);
    v5 = a2;
    v80 = v85;
  }
  else
  {
    v79 = ((__int64 (__fastcall *)(_QWORD *, const char *, __int64 (__fastcall *)(__int64), __int64, __int64, const char *))v11)(
            v9,
            a2,
            v11,
            a4,
            a5,
            a2);
    v5 = a2;
    v80 = v79;
  }
  v13 = v6 & v80;
  if ( !v13 )
    goto LABEL_3;
  v81 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
  if ( v81 != sub_16CD290 )
  {
    srcc = v5;
    ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v81)(v9, 8, 1, 0);
    v5 = srcc;
  }
LABEL_4:
  if ( v5 && *v5 )
  {
    srca = v5;
    v29 = strlen(v5);
    v30 = (_WORD *)v9[3];
    v31 = v29;
    v32 = v9[2] - (_QWORD)v30;
    if ( v31 > v32 )
    {
      v91 = sub_16E7EE0(v9, srca, v31);
      v30 = *(_WORD **)(v91 + 24);
      v33 = (_QWORD *)v91;
      v32 = *(_QWORD *)(v91 + 16) - (_QWORD)v30;
    }
    else
    {
      v33 = v9;
      if ( v31 )
      {
        memcpy(v30, srca, v31);
        v93 = v9[2];
        v33 = v9;
        v30 = (_WORD *)(v31 + v9[3]);
        v9[3] = v30;
        v32 = v93 - (_QWORD)v30;
      }
    }
    if ( v32 <= 1 )
    {
      sub_16E7EE0(v33, ": ", 2);
    }
    else
    {
      *v30 = 8250;
      v33[3] += 2LL;
    }
  }
  if ( *(_QWORD *)(a1 + 24) )
  {
    if ( (unsigned int)sub_2241AC0(a1 + 16, "-") )
    {
      sub_16E7EE0(v9, *(const char **)(a1 + 16), *(_QWORD *)(a1 + 24));
      v34 = (_BYTE *)v9[3];
    }
    else
    {
      v84 = v9[3];
      if ( (unsigned __int64)(v9[2] - v84) <= 6 )
      {
        sub_16E7EE0(v9, "<stdin>", 7);
        v34 = (_BYTE *)v9[3];
      }
      else
      {
        *(_DWORD *)v84 = 1685353276;
        *(_WORD *)(v84 + 4) = 28265;
        *(_BYTE *)(v84 + 6) = 62;
        v34 = (_BYTE *)(v9[3] + 7LL);
        v9[3] = v34;
      }
    }
    if ( *(_DWORD *)(a1 + 48) != -1 )
    {
      if ( v9[2] <= (unsigned __int64)v34 )
      {
        v35 = sub_16E7DE0(v9, 58);
      }
      else
      {
        v35 = (__int64)v9;
        v9[3] = v34 + 1;
        *v34 = 58;
      }
      sub_16E7AB0(v35, *(int *)(a1 + 48));
      if ( *(_DWORD *)(a1 + 52) != -1 )
      {
        v36 = (_BYTE *)v9[3];
        if ( (unsigned __int64)v36 >= v9[2] )
        {
          v37 = sub_16E7DE0(v9, 58);
        }
        else
        {
          v37 = (__int64)v9;
          v9[3] = v36 + 1;
          *v36 = 58;
        }
        sub_16E7AB0(v37, *(_DWORD *)(a1 + 52) + 1);
      }
      v34 = (_BYTE *)v9[3];
    }
    if ( v9[2] - (_QWORD)v34 <= 1u )
    {
      sub_16E7EE0(v9, ": ", 2);
    }
    else
    {
      *(_WORD *)v34 = 8250;
      v9[3] += 2LL;
    }
  }
  if ( v7 )
  {
    v14 = *(_DWORD *)(a1 + 56);
    if ( v14 == 2 )
    {
      if ( v13 )
      {
        v89 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
        if ( v89 != sub_16CD290 )
          ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v89)(v9, 4, 1, 0);
      }
      v90 = (_QWORD *)v9[3];
      if ( v9[2] - (_QWORD)v90 <= 7u )
      {
        sub_16E7EE0(v9, "remark: ", 8);
      }
      else
      {
        *v90 = 0x203A6B72616D6572LL;
        v9[3] += 8LL;
      }
    }
    else if ( v14 > 2 )
    {
      if ( v14 == 3 )
      {
        if ( v13 )
        {
          v24 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
          if ( v24 != sub_16CD290 )
            ((void (__fastcall *)(_QWORD *, _QWORD, __int64, _QWORD))v24)(v9, 0, 1, 0);
        }
        v25 = v9[3];
        if ( (unsigned __int64)(v9[2] - v25) <= 5 )
        {
          sub_16E7EE0(v9, "note: ", 6);
        }
        else
        {
          *(_DWORD *)v25 = 1702129518;
          *(_WORD *)(v25 + 4) = 8250;
          v9[3] += 6LL;
        }
      }
    }
    else if ( v14 )
    {
      if ( v13 )
      {
        v15 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
        if ( v15 != sub_16CD290 )
          ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v15)(v9, 5, 1, 0);
      }
      v16 = v9[3];
      if ( (unsigned __int64)(v9[2] - v16) <= 8 )
      {
        sub_16E7EE0(v9, "warning: ", 9);
      }
      else
      {
        *(_BYTE *)(v16 + 8) = 32;
        *(_QWORD *)v16 = 0x3A676E696E726177LL;
        v9[3] += 9LL;
      }
    }
    else
    {
      if ( v13 )
      {
        v82 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
        if ( v82 != sub_16CD290 )
          ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v82)(v9, 1, 1, 0);
      }
      v83 = v9[3];
      if ( (unsigned __int64)(v9[2] - v83) <= 6 )
      {
        sub_16E7EE0(v9, "error: ", 7);
      }
      else
      {
        *(_DWORD *)v83 = 1869771365;
        *(_WORD *)(v83 + 4) = 14962;
        *(_BYTE *)(v83 + 6) = 32;
        v9[3] += 7LL;
      }
    }
    if ( v13 )
    {
      v26 = *v9;
      v27 = *(__int64 (__fastcall **)(__int64))(*v9 + 24LL);
      if ( v27 != sub_16CD2A0 )
      {
        v27((__int64)v9);
        v26 = *v9;
      }
      v28 = *(__int64 (__fastcall **)(__int64))(v26 + 16);
      if ( v28 != sub_16CD290 )
        ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v28)(v9, 8, 1, 0);
    }
  }
  v17 = sub_16E7EE0(v9, *(const char **)(a1 + 64), *(_QWORD *)(a1 + 72));
  result = *(__int64 (__fastcall **)(__int64))(v17 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v17 + 16) )
  {
    result = (__int64 (__fastcall *)(__int64))sub_16E7DE0(v17, 10);
  }
  else
  {
    *(_QWORD *)(v17 + 24) = (char *)result + 1;
    *(_BYTE *)result = 10;
  }
  if ( v13 )
  {
    result = *(__int64 (__fastcall **)(__int64))(*v9 + 24LL);
    if ( result != sub_16CD2A0 )
      result = (__int64 (__fastcall *)(__int64))result((__int64)v9);
  }
  if ( *(_DWORD *)(a1 + 48) == -1 || *(_DWORD *)(a1 + 52) == -1 )
    return result;
  v20 = *(unsigned __int8 **)(a1 + 96);
  src = *(_QWORD *)(a1 + 104);
  v21 = (char *)&v20[src];
  v22 = src;
  if ( src >> 2 > 0 )
  {
    v23 = *(char **)(a1 + 96);
    while ( *v23 >= 0 )
    {
      if ( v23[1] < 0 )
      {
        ++v23;
        break;
      }
      if ( v23[2] < 0 )
      {
        v23 += 2;
        break;
      }
      if ( v23[3] < 0 )
      {
        v23 += 3;
        break;
      }
      v23 += 4;
      if ( &v20[4 * (src >> 2)] == (unsigned __int8 *)v23 )
      {
        v22 = v21 - v23;
        goto LABEL_69;
      }
    }
LABEL_31:
    if ( v21 != v23 )
      return (__int64 (__fastcall *)(__int64))sub_16CD420((__int64)v9, v20, src);
    goto LABEL_72;
  }
  v23 = *(char **)(a1 + 96);
LABEL_69:
  if ( v22 != 2 )
  {
    if ( v22 != 3 )
    {
      if ( v22 != 1 )
        goto LABEL_72;
      goto LABEL_170;
    }
    if ( *v23 < 0 )
      goto LABEL_31;
    ++v23;
  }
  if ( *v23 < 0 )
    goto LABEL_31;
  ++v23;
LABEL_170:
  if ( *v23 < 0 )
    goto LABEL_31;
LABEL_72:
  v111 = v113;
  sub_2240A50(&v111, src + 1, 32, v21, v18);
  v38 = (__int64)(*(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 128)) >> 3;
  if ( (_DWORD)v38 )
  {
    v39 = (unsigned int)(v38 - 1);
    v40 = 0;
    v41 = *(_QWORD *)(a1 + 128);
    v42 = 8 * v39;
    while ( 1 )
    {
      v43 = (unsigned int *)(v40 + v41);
      v44 = v43[1];
      v45 = *v43;
      if ( v44 > v112 )
        v44 = v112;
      v46 = v44 - v45;
      if ( v46 )
      {
        memset((char *)v111 + v45, 126, v46);
        if ( v42 == v40 )
          break;
      }
      else if ( v42 == v40 )
      {
        break;
      }
      v41 = *(_QWORD *)(a1 + 128);
      v40 += 8;
    }
  }
  v47 = *(_QWORD *)(a1 + 104);
  v48 = *(_QWORD *)(a1 + 8);
  LOBYTE(v116[0]) = 0;
  v49 = *(unsigned int *)(a1 + 160);
  v114 = v116;
  v50 = *(int *)(a1 + 52);
  v115 = 0;
  if ( v49 )
  {
    v103 = v13;
    v102 = a1;
    v101 = v9;
    v51 = v48 - v50;
    v105 = v48 - v50 + v47;
    v52 = *(unsigned __int64 **)(a1 + 152);
    v104 = v47;
    v53 = 0;
    v54 = (__int64 *)&v52[6 * v49];
    do
    {
      v62 = v52[3];
      v110[0] = v52[2];
      v110[1] = v62;
      if ( sub_16D23E0(v110, "\n\r\t", 3, 0) == -1 )
      {
        v92 = v52[1];
        v63 = *v52;
        if ( v51 <= v92 && v105 >= v63 )
        {
          if ( v51 <= v63 )
          {
            v55 = v63 - v51;
            v56 = (unsigned int)(v63 - v51);
            v57 = (unsigned int)v56;
          }
          else
          {
            v57 = 0;
            v56 = 0;
            v55 = 0;
          }
          v58 = v56;
          if ( v56 < v53 )
          {
            v58 = (unsigned int)(v53 + 1);
            v55 = v53 + 1;
          }
          v59 = v52[3];
          v53 = (unsigned int)(v55 + v59);
          if ( v53 > v115 )
          {
            v94 = v52[1];
            v96 = v57;
            v98 = v56;
            v100 = v58;
            sub_22410F0(&v114, v53, 32);
            v59 = v52[3];
            v92 = v94;
            v57 = v96;
            v56 = v98;
            v58 = v100;
          }
          if ( v59 )
          {
            v95 = v92;
            v97 = v57;
            v99 = v56;
            memmove((char *)v114 + v58, (const void *)v52[2], v59);
            v92 = v95;
            v57 = v97;
            v56 = v99;
          }
          v60 = v92 - v51;
          if ( v105 <= v92 )
            v60 = v104;
          v61 = v60 - v57;
          if ( v61 )
            memset((char *)v111 + v56, 126, v61);
        }
      }
      v52 += 6;
    }
    while ( v54 != (__int64 *)v52 );
    a1 = v102;
    v13 = v103;
    v9 = v101;
    v50 = *(int *)(v102 + 52);
  }
  if ( (unsigned int)v50 > (unsigned __int64)src )
    v50 = src;
  *((_BYTE *)v111 + v50) = 94;
  v64 = sub_2241A80(&v111, 32, -1);
  if ( v64 + 1 > v112 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::erase");
  v112 = v64 + 1;
  *((_BYTE *)v111 + v64 + 1) = 0;
  v65 = *(_QWORD *)(a1 + 96);
  sub_16CD420((__int64)v9, (unsigned __int8 *)v65, *(_QWORD *)(a1 + 104));
  if ( v13 )
  {
    v88 = *(__int64 (__fastcall **)(__int64))(*v9 + 16LL);
    if ( v88 != sub_16CD290 )
    {
      v65 = 2;
      ((void (__fastcall *)(_QWORD *, __int64, __int64, _QWORD))v88)(v9, 2, 1, 0);
    }
  }
  v66 = 0;
  v67 = 0;
  v68 = (unsigned int)v112;
  if ( (_DWORD)v112 )
  {
    srcb = v13;
    v69 = (unsigned int)v112;
    do
    {
      if ( *(_QWORD *)(a1 + 104) > v66 && *(_BYTE *)(*(_QWORD *)(a1 + 96) + v66) == 9 )
      {
        do
        {
          v65 = *((unsigned __int8 *)v111 + v66);
          v71 = (_BYTE *)v9[3];
          if ( (unsigned __int64)v71 < v9[2] )
          {
            v9[3] = v71 + 1;
            *v71 = v65;
          }
          else
          {
            sub_16E7DE0(v9, v65);
          }
          ++v67;
        }
        while ( (v67 & 7) != 0 );
      }
      else
      {
        v65 = *((unsigned __int8 *)v111 + v66);
        v70 = (_BYTE *)v9[3];
        if ( (unsigned __int64)v70 < v9[2] )
        {
          v9[3] = v70 + 1;
          *v70 = v65;
        }
        else
        {
          sub_16E7DE0(v9, v65);
        }
        ++v67;
      }
      ++v66;
    }
    while ( v66 != v69 );
    v13 = srcb;
  }
  result = (__int64 (__fastcall *)(__int64))v9[3];
  if ( (unsigned __int64)result >= v9[2] )
  {
    v65 = 10;
    result = (__int64 (__fastcall *)(__int64))sub_16E7DE0(v9, 10);
  }
  else
  {
    v72 = (char *)result + 1;
    v9[3] = (char *)result + 1;
    *(_BYTE *)result = 10;
  }
  if ( v13 )
  {
    result = *(__int64 (__fastcall **)(__int64))(*v9 + 24LL);
    if ( result != sub_16CD2A0 )
      result = (__int64 (__fastcall *)(__int64))((__int64 (__fastcall *)(_QWORD *, __int64, char *, __int64))result)(
                                                  v9,
                                                  v65,
                                                  v72,
                                                  v68);
  }
  v73 = v115;
  v74 = 0;
  v75 = 0;
  if ( v115 )
  {
    do
    {
      v76 = v114;
      if ( *(_QWORD *)(a1 + 104) > v75 && *(_BYTE *)(*(_QWORD *)(a1 + 96) + v75) == 9 )
      {
        do
        {
          v86 = *((unsigned __int8 *)v76 + v75);
          v87 = (_BYTE *)v9[3];
          if ( (unsigned __int64)v87 < v9[2] )
          {
            v9[3] = v87 + 1;
            *v87 = v86;
          }
          else
          {
            sub_16E7DE0(v9, v86);
          }
          v76 = v114;
          ++v74;
          v75 += *((_BYTE *)v114 + v75) != 32;
        }
        while ( (v74 & 7) != 0 && v75 != v73 );
      }
      else
      {
        v77 = *((unsigned __int8 *)v114 + v75);
        v78 = (_BYTE *)v9[3];
        if ( (unsigned __int64)v78 < v9[2] )
        {
          v9[3] = v78 + 1;
          *v78 = v77;
        }
        else
        {
          sub_16E7DE0(v9, v77);
        }
        ++v74;
      }
      ++v75;
    }
    while ( v75 < v73 );
    result = (__int64 (__fastcall *)(__int64))v9[3];
    if ( (unsigned __int64)result >= v9[2] )
    {
      result = (__int64 (__fastcall *)(__int64))sub_16E7DE0(v9, 10);
    }
    else
    {
      v9[3] = (char *)result + 1;
      *(_BYTE *)result = 10;
    }
  }
  if ( v114 != v116 )
    result = (__int64 (__fastcall *)(__int64))j_j___libc_free_0(v114, v116[0] + 1LL);
  if ( v111 != v113 )
    return (__int64 (__fastcall *)(__int64))j_j___libc_free_0(v111, v113[0] + 1LL);
  return result;
}
