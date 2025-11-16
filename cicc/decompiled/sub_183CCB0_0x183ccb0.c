// Function: sub_183CCB0
// Address: 0x183ccb0
//
void __fastcall sub_183CCB0(unsigned __int64 a1, __int64 a2, _DWORD *a3, char a4, int a5, int a6)
{
  unsigned __int64 *v8; // rbx
  unsigned __int8 v9; // al
  unsigned __int64 v10; // rdx
  _DWORD *v11; // r12
  unsigned __int64 v12; // rcx
  _BYTE *v13; // rax
  _BYTE *v14; // rsi
  int v15; // r14d
  size_t v16; // r8
  _DWORD *v17; // r13
  unsigned __int64 v18; // r14
  size_t v19; // r8
  __int64 v20; // rax
  void *v21; // r9
  bool v22; // r14
  void *v23; // rax
  unsigned __int64 v24; // rbx
  _BYTE *v25; // rax
  int v26; // r13d
  size_t v27; // r9
  signed __int64 v28; // r14
  __int64 v29; // rax
  void *v30; // r8
  unsigned int v31; // esi
  char v32; // al
  _QWORD *v33; // rax
  size_t v34; // rcx
  unsigned __int64 v35; // r12
  int v36; // ebx
  __int64 v37; // rax
  __int64 v38; // rsi
  int v39; // r9d
  unsigned int v40; // ecx
  __int64 v41; // r13
  _BYTE *v42; // rax
  unsigned __int64 v43; // r14
  __int64 v44; // rax
  size_t v45; // r13
  _QWORD *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rcx
  int v49; // r10d
  unsigned __int64 v50; // rsi
  __int64 v51; // r8
  _BYTE *v52; // rax
  unsigned __int64 v53; // r13
  _BYTE *v54; // rax
  size_t v55; // r9
  void *v56; // r13
  _BYTE *v57; // rax
  size_t v58; // r9
  void *v59; // r10
  void *v60; // rax
  bool v61; // al
  _BYTE *v62; // rax
  int v63; // ebx
  size_t v64; // r8
  signed __int64 v65; // r13
  void *v66; // r9
  unsigned int v67; // eax
  _BYTE *v68; // rax
  unsigned __int64 v69; // r13
  size_t v70; // r14
  int v71; // eax
  size_t v72; // rdx
  int v73; // eax
  int v74; // eax
  unsigned int v75; // eax
  int v76; // r8d
  int v77; // r9d
  size_t v78; // r13
  unsigned int v79; // ebx
  _BYTE *v80; // rax
  __int64 v81; // rax
  size_t v82; // r13
  size_t v83; // rdx
  int v84; // eax
  size_t na; // [rsp+8h] [rbp-88h]
  size_t n; // [rsp+8h] [rbp-88h]
  size_t v87; // [rsp+10h] [rbp-80h]
  signed __int64 v88; // [rsp+10h] [rbp-80h]
  size_t v89; // [rsp+18h] [rbp-78h]
  size_t v90; // [rsp+18h] [rbp-78h]
  size_t v91; // [rsp+18h] [rbp-78h]
  size_t v92; // [rsp+18h] [rbp-78h]
  size_t v93; // [rsp+18h] [rbp-78h]
  char *v94; // [rsp+20h] [rbp-70h]
  unsigned __int64 v95; // [rsp+28h] [rbp-68h]
  size_t v96; // [rsp+28h] [rbp-68h]
  int v97; // [rsp+28h] [rbp-68h]
  size_t v98; // [rsp+28h] [rbp-68h]
  char *v99; // [rsp+28h] [rbp-68h]
  void *v100; // [rsp+28h] [rbp-68h]
  unsigned __int64 v101; // [rsp+28h] [rbp-68h]
  char *v102; // [rsp+30h] [rbp-60h]
  __int64 v103; // [rsp+30h] [rbp-60h]
  unsigned __int64 v104; // [rsp+30h] [rbp-60h]
  int v105; // [rsp+30h] [rbp-60h]
  int v106; // [rsp+30h] [rbp-60h]
  bool v107; // [rsp+30h] [rbp-60h]
  bool v108; // [rsp+30h] [rbp-60h]
  unsigned __int64 v109; // [rsp+30h] [rbp-60h]
  size_t v110; // [rsp+30h] [rbp-60h]
  unsigned __int64 v111; // [rsp+30h] [rbp-60h]
  unsigned __int64 v112; // [rsp+30h] [rbp-60h]
  void *v113; // [rsp+30h] [rbp-60h]
  unsigned __int64 v114; // [rsp+30h] [rbp-60h]
  size_t v115; // [rsp+30h] [rbp-60h]
  unsigned int v116; // [rsp+38h] [rbp-58h]
  unsigned int v117; // [rsp+38h] [rbp-58h]
  void *v118; // [rsp+38h] [rbp-58h]
  void *v119; // [rsp+38h] [rbp-58h]
  unsigned int v120; // [rsp+40h] [rbp-50h] BYREF
  void *s1; // [rsp+48h] [rbp-48h]
  char *v122; // [rsp+50h] [rbp-40h]
  char *v123; // [rsp+58h] [rbp-38h]

  v8 = (unsigned __int64 *)a1;
  v9 = *(_BYTE *)(a2 + 16);
  if ( v9 == 26 )
  {
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 1 )
    {
LABEL_27:
      **(_BYTE **)a3 = 1;
      return;
    }
    v10 = *(_QWORD *)(a2 - 72) & 0xFFFFFFFFFFFFFFF9LL;
    if ( a4 )
    {
      a1 = (unsigned __int64)&v120;
      sub_183C910((__int64)&v120, v8, v10);
      v11 = s1;
      v12 = *v8;
      v116 = v120;
      v102 = v122;
      v94 = v123;
      goto LABEL_5;
    }
    v37 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v37 )
    {
      v38 = *(_QWORD *)(a1 + 16);
      v39 = 1;
      v40 = (v37 - 1) & (v10 ^ (v10 >> 9));
      v41 = v38 + 40LL * v40;
      a1 = *(_QWORD *)v41;
      if ( v10 == *(_QWORD *)v41 )
      {
LABEL_42:
        if ( v41 != v38 + 40 * v37 )
        {
          v14 = *(_BYTE **)(v41 + 16);
          v116 = *(_DWORD *)(v41 + 8);
          v42 = *(_BYTE **)(v41 + 24);
          v43 = v42 - v14;
          if ( v42 == v14 )
          {
            v45 = 0;
            v11 = 0;
          }
          else
          {
            if ( v43 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_173;
            v44 = sub_22077B0(v43);
            v14 = *(_BYTE **)(v41 + 16);
            v11 = (_DWORD *)v44;
            v42 = *(_BYTE **)(v41 + 24);
            v45 = v42 - v14;
          }
          a1 = (unsigned __int64)v11 + v43;
          v94 = (char *)v11 + v43;
          if ( v42 != v14 )
          {
            a1 = (unsigned __int64)v11;
            memmove(v11, v14, v45);
          }
          v12 = *v8;
          v102 = (char *)v11 + v45;
          goto LABEL_5;
        }
      }
      else
      {
        while ( a1 != -2 )
        {
          v40 = (v37 - 1) & (v39 + v40);
          v41 = v38 + 40LL * v40;
          a1 = *(_QWORD *)v41;
          if ( v10 == *(_QWORD *)v41 )
            goto LABEL_42;
          ++v39;
        }
      }
    }
    v12 = *v8;
    v14 = *(_BYTE **)(*v8 + 80);
    v116 = *(_DWORD *)(*v8 + 72);
    v68 = *(_BYTE **)(*v8 + 88);
    v69 = v68 - v14;
    if ( v68 == v14 )
    {
      v70 = 0;
      v11 = 0;
    }
    else
    {
      v111 = *v8;
      if ( v69 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      v11 = (_DWORD *)sub_22077B0(v69);
      v68 = *(_BYTE **)(v111 + 88);
      v14 = *(_BYTE **)(v111 + 80);
      v12 = *v8;
      v70 = v68 - v14;
    }
    a1 = (unsigned __int64)v11 + v69;
    v94 = (char *)v11 + v69;
    if ( v14 != v68 )
    {
      a1 = (unsigned __int64)v11;
      v112 = v12;
      memmove(v11, v14, v70);
      v12 = v112;
    }
    v102 = (char *)v11 + v70;
LABEL_5:
    v13 = *(_BYTE **)(v12 + 56);
    v14 = *(_BYTE **)(v12 + 48);
    v95 = v12;
    v15 = *(_DWORD *)(v12 + 40);
    v16 = v13 - v14;
    v89 = v13 - v14;
    if ( v13 == v14 )
    {
      v17 = 0;
    }
    else
    {
      if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      a1 = v13 - v14;
      v17 = (_DWORD *)sub_22077B0(v16);
      v13 = *(_BYTE **)(v95 + 56);
      v14 = *(_BYTE **)(v95 + 48);
      v16 = v13 - v14;
    }
    if ( v13 != v14 )
    {
      a1 = (unsigned __int64)v17;
      v96 = v16;
      memmove(v17, v14, v16);
      v16 = v96;
    }
    if ( v15 == v116 && v102 - (char *)v11 == v16 )
    {
      if ( !v16 )
      {
        if ( !v17 )
        {
LABEL_99:
          *(_BYTE *)(*(_QWORD *)a3 + 1LL) = 1;
          **(_BYTE **)a3 = 1;
          goto LABEL_100;
        }
LABEL_98:
        j_j___libc_free_0(v17, v89);
        goto LABEL_99;
      }
      a1 = (unsigned __int64)v11;
      if ( !memcmp(v11, v17, v16) )
        goto LABEL_98;
    }
    v18 = *v8;
    v14 = *(_BYTE **)(*v8 + 80);
    v97 = *(_DWORD *)(*v8 + 72);
    v19 = *(_QWORD *)(*v8 + 88) - (_QWORD)v14;
    v87 = v19;
    if ( v19 )
    {
      if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      v20 = sub_22077B0(v19);
      v14 = *(_BYTE **)(v18 + 80);
      v21 = (void *)v20;
      v19 = *(_QWORD *)(v18 + 88) - (_QWORD)v14;
      if ( *(_BYTE **)(v18 + 88) != v14 )
      {
LABEL_14:
        na = v19;
        v22 = 0;
        v23 = memmove(v21, v14, v19);
        a1 = v116;
        v19 = na;
        v21 = v23;
        if ( v97 != v116 )
        {
LABEL_15:
          a1 = (unsigned __int64)v21;
          j_j___libc_free_0(v21, v87);
LABEL_16:
          if ( v17 )
          {
            a1 = (unsigned __int64)v17;
            j_j___libc_free_0(v17, v89);
          }
          if ( v22 )
            goto LABEL_99;
          v24 = *v8;
          v25 = *(_BYTE **)(v24 + 24);
          v14 = *(_BYTE **)(v24 + 16);
          v26 = *(_DWORD *)(v24 + 8);
          v27 = v25 - v14;
          v28 = v25 - v14;
          if ( v25 == v14 )
          {
            v30 = 0;
LABEL_22:
            if ( v25 == v14 )
            {
              if ( v26 == v116 )
              {
                v72 = v102 - (char *)v11;
                if ( v27 == v102 - (char *)v11 )
                  goto LABEL_115;
              }
              if ( !v30 )
              {
LABEL_25:
                if ( !v11 )
                {
LABEL_26:
                  *(_BYTE *)(*(_QWORD *)a3 + 1LL) = 1;
                  goto LABEL_27;
                }
LABEL_118:
                j_j___libc_free_0(v11, v94 - (char *)v11);
                goto LABEL_26;
              }
            }
            else
            {
              v98 = v27;
              v30 = memmove(v30, v14, v27);
              if ( v26 == v116 )
              {
                v72 = v102 - (char *)v11;
                if ( v98 == v102 - (char *)v11 )
                {
LABEL_115:
                  if ( v72 )
                  {
                    v118 = v30;
                    v73 = memcmp(v11, v30, v72);
                    v30 = v118;
                    if ( v73 )
                    {
                      j_j___libc_free_0(v118, v28);
                      goto LABEL_118;
                    }
                  }
                  else if ( !v30 )
                  {
                    goto LABEL_100;
                  }
                  j_j___libc_free_0(v30, v28);
LABEL_100:
                  if ( v11 )
                    j_j___libc_free_0(v11, v94 - (char *)v11);
                  return;
                }
              }
            }
            j_j___libc_free_0(v30, v28);
            goto LABEL_25;
          }
          if ( v27 <= 0x7FFFFFFFFFFFFFF8LL )
          {
            v29 = sub_22077B0(v27);
            v14 = *(_BYTE **)(v24 + 16);
            v30 = (void *)v29;
            v25 = *(_BYTE **)(v24 + 24);
            v27 = v25 - v14;
            goto LABEL_22;
          }
LABEL_173:
          sub_4261EA(a1, v14, v10);
        }
LABEL_103:
        if ( v19 == v102 - (char *)v11 )
        {
          if ( v19 )
          {
            v100 = v21;
            v71 = memcmp(v11, v21, v19);
            v21 = v100;
            v22 = v71 == 0;
            goto LABEL_15;
          }
          v22 = 1;
LABEL_52:
          if ( !v21 )
            goto LABEL_16;
          goto LABEL_15;
        }
LABEL_51:
        v22 = 0;
        goto LABEL_52;
      }
    }
    else
    {
      v21 = 0;
      if ( *(_BYTE **)(*v8 + 88) != v14 )
        goto LABEL_14;
    }
    a1 = v116;
    if ( v97 != v116 )
      goto LABEL_51;
    goto LABEL_103;
  }
  v31 = v9 - 24;
  if ( v31 > 6 )
  {
    if ( (unsigned int)v9 - 32 <= 2 )
      goto LABEL_37;
  }
  else if ( v31 > 4 )
  {
    goto LABEL_37;
  }
  if ( v9 != 28 )
  {
    v32 = *(_BYTE *)(a2 + 23) & 0x40;
    if ( a4 )
    {
      if ( v32 )
        v33 = *(_QWORD **)(a2 - 8);
      else
        v33 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      a1 = (unsigned __int64)&v120;
      sub_183C910((__int64)&v120, v8, *v33 & 0xFFFFFFFFFFFFFFF9LL);
      v11 = s1;
      v34 = *v8;
      v117 = v120;
      v99 = v122;
      v94 = v123;
      goto LABEL_65;
    }
    if ( v32 )
      v46 = *(_QWORD **)(a2 - 8);
    else
      v46 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v47 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v47 )
    {
      v48 = *(_QWORD *)(a1 + 16);
      v49 = 1;
      v50 = *v46 & 0xFFFFFFFFFFFFFFF9LL;
      v10 = ((_DWORD)v47 - 1) & ((unsigned int)v50 ^ (unsigned int)(*v46 >> 9));
      v51 = v48 + 40LL * (unsigned int)v10;
      a1 = *(_QWORD *)v51;
      if ( v50 == *(_QWORD *)v51 )
      {
LABEL_58:
        if ( v51 != v48 + 40 * v47 )
        {
          v14 = *(_BYTE **)(v51 + 16);
          v103 = v51;
          v117 = *(_DWORD *)(v51 + 8);
          v52 = *(_BYTE **)(v51 + 24);
          v53 = v52 - v14;
          if ( v52 == v14 )
          {
            v10 = 0;
            v11 = 0;
          }
          else
          {
            if ( v53 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_173;
            v11 = (_DWORD *)sub_22077B0(v53);
            v52 = *(_BYTE **)(v103 + 24);
            v14 = *(_BYTE **)(v103 + 16);
            v10 = v52 - v14;
          }
          a1 = (unsigned __int64)v11 + v53;
          v94 = (char *)v11 + v53;
          if ( v14 != v52 )
          {
            a1 = (unsigned __int64)v11;
            v104 = v10;
            memmove(v11, v14, v10);
            v10 = v104;
          }
          v34 = *v8;
          v99 = (char *)v11 + v10;
          goto LABEL_65;
        }
      }
      else
      {
        while ( a1 != -2 )
        {
          v10 = ((_DWORD)v47 - 1) & (unsigned int)(v49 + v10);
          v51 = v48 + 40LL * (unsigned int)v10;
          a1 = *(_QWORD *)v51;
          if ( v50 == *(_QWORD *)v51 )
            goto LABEL_58;
          ++v49;
        }
      }
    }
    v34 = *v8;
    v14 = *(_BYTE **)(*v8 + 80);
    v117 = *(_DWORD *)(*v8 + 72);
    v80 = *(_BYTE **)(*v8 + 88);
    v10 = v80 - v14;
    if ( v80 == v14 )
    {
      v82 = 0;
      v11 = 0;
    }
    else
    {
      v101 = *v8;
      if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      v114 = *(_QWORD *)(*v8 + 88) - (_QWORD)v14;
      v81 = sub_22077B0(v10);
      v10 = v114;
      v11 = (_DWORD *)v81;
      v80 = *(_BYTE **)(v101 + 88);
      v14 = *(_BYTE **)(v101 + 80);
      v34 = *v8;
      v82 = v80 - v14;
    }
    a1 = (unsigned __int64)v11 + v10;
    v94 = (char *)v11 + v10;
    if ( v14 != v80 )
    {
      a1 = (unsigned __int64)v11;
      v115 = v34;
      memmove(v11, v14, v82);
      v34 = v115;
    }
    v99 = (char *)v11 + v82;
LABEL_65:
    v14 = *(_BYTE **)(v34 + 48);
    v90 = v34;
    v105 = *(_DWORD *)(v34 + 40);
    v54 = *(_BYTE **)(v34 + 56);
    v55 = v54 - v14;
    v88 = v54 - v14;
    if ( v54 == v14 )
    {
      v56 = 0;
    }
    else
    {
      if ( v55 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      v56 = (void *)sub_22077B0(v55);
      v54 = *(_BYTE **)(v90 + 56);
      v14 = *(_BYTE **)(v90 + 48);
      v55 = v54 - v14;
    }
    if ( v54 != v14 )
    {
      v91 = v55;
      memmove(v56, v14, v55);
      v55 = v91;
    }
    a1 = v117;
    if ( v105 == v117 && v99 - (char *)v11 == v55 )
    {
      if ( !v55 )
      {
        if ( !v56 )
          goto LABEL_137;
        goto LABEL_136;
      }
      a1 = (unsigned __int64)v11;
      if ( !memcmp(v11, v56, v55) )
      {
LABEL_136:
        j_j___libc_free_0(v56, v88);
LABEL_137:
        v75 = sub_15F4D60(a2);
        a3[2] = 0;
        v78 = v75;
        v79 = v75;
        if ( a3[3] < v75 )
          sub_16CD150((__int64)a3, a3 + 4, v75, 1, v76, v77);
        a3[2] = v79;
        if ( v78 )
          memset(*(void **)a3, 1, v78);
        goto LABEL_100;
      }
    }
    v10 = *v8;
    v14 = *(_BYTE **)(*v8 + 80);
    v92 = *v8;
    v106 = *(_DWORD *)(*v8 + 72);
    v57 = *(_BYTE **)(*v8 + 88);
    v58 = v57 - v14;
    n = v57 - v14;
    if ( v57 == v14 )
    {
      v59 = 0;
    }
    else
    {
      if ( v58 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_173;
      a1 = *(_QWORD *)(*v8 + 88) - (_QWORD)v14;
      v59 = (void *)sub_22077B0(v58);
      v57 = *(_BYTE **)(v92 + 88);
      v14 = *(_BYTE **)(v92 + 80);
      v58 = v57 - v14;
    }
    if ( v57 == v14 )
    {
      if ( v106 != v117 )
        goto LABEL_126;
    }
    else
    {
      v93 = v58;
      v60 = memmove(v59, v14, v58);
      a1 = v117;
      v58 = v93;
      v59 = v60;
      v61 = 0;
      if ( v106 != v117 )
        goto LABEL_76;
    }
    if ( v99 - (char *)v11 == v58 )
    {
      if ( v58 )
      {
        v113 = v59;
        v74 = memcmp(v11, v59, v58);
        v59 = v113;
        v61 = v74 == 0;
        goto LABEL_76;
      }
      v61 = 1;
      goto LABEL_127;
    }
LABEL_126:
    v61 = 0;
LABEL_127:
    if ( !v59 )
    {
LABEL_77:
      if ( v56 )
      {
        a1 = (unsigned __int64)v56;
        v108 = v61;
        j_j___libc_free_0(v56, v88);
        v61 = v108;
      }
      if ( v61 )
        goto LABEL_137;
      v10 = *v8;
      v62 = *(_BYTE **)(*v8 + 24);
      v14 = *(_BYTE **)(*v8 + 16);
      v109 = *v8;
      v63 = *(_DWORD *)(*v8 + 8);
      v64 = v62 - v14;
      v65 = v62 - v14;
      if ( v62 == v14 )
      {
        v66 = 0;
      }
      else
      {
        if ( v64 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_173;
        v66 = (void *)sub_22077B0(v64);
        v62 = *(_BYTE **)(v109 + 24);
        v14 = *(_BYTE **)(v109 + 16);
        v64 = v62 - v14;
      }
      if ( v62 == v14 )
      {
        if ( v63 == v117 )
        {
          v83 = v99 - (char *)v11;
          if ( v64 == v99 - (char *)v11 )
          {
LABEL_157:
            if ( v83 )
            {
              v119 = v66;
              v84 = memcmp(v11, v66, v83);
              v66 = v119;
              if ( v84 )
              {
                j_j___libc_free_0(v119, v65);
LABEL_160:
                j_j___libc_free_0(v11, v94 - (char *)v11);
LABEL_87:
                v67 = sub_15F4D60(a2);
                a3[2] = 0;
                v35 = v67;
                v36 = v67;
                if ( a3[3] >= v67 )
                  goto LABEL_38;
                goto LABEL_88;
              }
            }
            else if ( !v66 )
            {
              goto LABEL_100;
            }
            j_j___libc_free_0(v66, v65);
            goto LABEL_100;
          }
        }
        if ( !v66 )
        {
LABEL_86:
          if ( !v11 )
            goto LABEL_87;
          goto LABEL_160;
        }
      }
      else
      {
        v110 = v64;
        v66 = memmove(v66, v14, v64);
        if ( v63 == v117 )
        {
          v83 = v99 - (char *)v11;
          if ( v110 == v99 - (char *)v11 )
            goto LABEL_157;
        }
      }
      j_j___libc_free_0(v66, v65);
      goto LABEL_86;
    }
LABEL_76:
    a1 = (unsigned __int64)v59;
    v107 = v61;
    j_j___libc_free_0(v59, n);
    v61 = v107;
    goto LABEL_77;
  }
LABEL_37:
  v35 = (unsigned int)a3[2];
  a3[2] = 0;
  v36 = v35;
  if ( a3[3] >= (unsigned int)v35 )
    goto LABEL_38;
LABEL_88:
  sub_16CD150((__int64)a3, a3 + 4, v35, 1, a5, a6);
LABEL_38:
  a3[2] = v36;
  if ( v35 )
    memset(*(void **)a3, 1, v35);
}
