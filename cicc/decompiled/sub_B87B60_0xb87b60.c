// Function: sub_B87B60
// Address: 0xb87b60
//
__int64 *__fastcall sub_B87B60(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 *result; // rax
  unsigned int v7; // r8d
  __int64 v8; // rcx
  __int64 v9; // r9
  int v10; // r11d
  __int64 *v11; // rax
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // rdi
  _QWORD *v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _QWORD *v19; // r12
  __int64 **v20; // rdi
  __int64 v21; // rax
  __int64 **v22; // rdx
  int v23; // ecx
  __int64 **v24; // rax
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r8
  _QWORD *v28; // rdi
  __int64 **v29; // rax
  __int64 v30; // rcx
  __int64 **v31; // rdx
  __int64 *v32; // rsi
  __int64 v33; // rax
  __int64 *v34; // r13
  __int64 v35; // rax
  __int64 *v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  _BYTE *v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // r11
  _QWORD *v45; // rax
  __int64 v46; // rdx
  _QWORD *v47; // r13
  __int64 v48; // r12
  _QWORD *v49; // rbx
  __int64 v50; // rsi
  _QWORD *v51; // rax
  _QWORD *v52; // r11
  _QWORD *v53; // r13
  char v54; // dl
  __int64 *v55; // rax
  __int64 *v56; // r12
  __int64 *v57; // rbx
  unsigned int v58; // eax
  __int64 v59; // rdx
  __int64 v60; // rdx
  char v61; // di
  __int64 *v62; // rax
  __int64 v63; // rcx
  __int64 *i; // rdx
  __int64 *v65; // rax
  unsigned int v66; // esi
  __int64 v67; // rcx
  __int64 v68; // r9
  unsigned int v69; // r8d
  _QWORD *v70; // rax
  __int64 v71; // rdi
  __int64 *v72; // rax
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  int v75; // eax
  int v76; // eax
  int v77; // edi
  int v78; // edi
  __int64 v79; // r10
  unsigned int v80; // esi
  __int64 v81; // r8
  int v82; // r14d
  _QWORD *v83; // r9
  _QWORD *v84; // rax
  int v85; // esi
  int v86; // esi
  __int64 v87; // r9
  _QWORD *v88; // r8
  unsigned int v89; // r14d
  int v90; // r10d
  __int64 v91; // rdi
  int v92; // edi
  int v93; // edx
  int v94; // ebx
  __int64 v95; // r8
  int v96; // ebx
  __int64 v97; // r11
  unsigned int v98; // esi
  int v99; // r9d
  __int64 *v100; // rdi
  int v101; // ebx
  int v102; // ebx
  __int64 v103; // r11
  int v104; // r9d
  unsigned int v105; // esi
  _QWORD *v106; // [rsp+8h] [rbp-178h]
  _QWORD *v107; // [rsp+8h] [rbp-178h]
  __int64 v108; // [rsp+10h] [rbp-170h]
  unsigned int v109; // [rsp+1Ch] [rbp-164h]
  _QWORD *v110; // [rsp+20h] [rbp-160h]
  int v111; // [rsp+20h] [rbp-160h]
  __int64 v112; // [rsp+20h] [rbp-160h]
  __int64 v113; // [rsp+20h] [rbp-160h]
  __int64 v114; // [rsp+20h] [rbp-160h]
  __int64 v115; // [rsp+20h] [rbp-160h]
  __int64 *v116; // [rsp+38h] [rbp-148h]
  _QWORD *v117; // [rsp+40h] [rbp-140h]
  __int64 v118; // [rsp+48h] [rbp-138h]
  __int64 *v119; // [rsp+50h] [rbp-130h]
  __int64 v120[2]; // [rsp+58h] [rbp-128h] BYREF
  __int64 *v121; // [rsp+68h] [rbp-118h] BYREF
  _BYTE *v122; // [rsp+70h] [rbp-110h] BYREF
  __int64 v123; // [rsp+78h] [rbp-108h]
  _BYTE v124[96]; // [rsp+80h] [rbp-100h] BYREF
  _BYTE *v125; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v126; // [rsp+E8h] [rbp-98h]
  _BYTE v127[144]; // [rsp+F0h] [rbp-90h] BYREF

  v4 = *(_QWORD *)(a4 + 8);
  v120[0] = a4;
  v109 = 0;
  if ( v4 )
    v109 = *(_DWORD *)(*(_QWORD *)(v4 + 24) + 384LL);
  result = &a2[a3];
  v119 = a2;
  v108 = a1 + 192;
  v116 = result;
  v118 = a1 + 224;
  if ( result == a2 )
    return result;
  do
  {
    v7 = *(_DWORD *)(a1 + 216);
    v8 = *v119;
    v121 = (__int64 *)*v119;
    if ( !v7 )
    {
      ++*(_QWORD *)(a1 + 192);
      goto LABEL_124;
    }
    v9 = *(_QWORD *)(a1 + 200);
    v10 = 1;
    v11 = 0;
    v12 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v8 != *v13 )
    {
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v11 )
          v11 = v13;
        v12 = (v7 - 1) & (v10 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_6;
        ++v10;
      }
      v92 = *(_DWORD *)(a1 + 208);
      if ( !v11 )
        v11 = v13;
      ++*(_QWORD *)(a1 + 192);
      v93 = v92 + 1;
      if ( 4 * (v92 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 212) - v93 > v7 >> 3 )
          goto LABEL_120;
        sub_B85070(v108, v7);
        v101 = *(_DWORD *)(a1 + 216);
        if ( !v101 )
        {
LABEL_164:
          ++*(_DWORD *)(a1 + 208);
          BUG();
        }
        v95 = (__int64)v121;
        v102 = v101 - 1;
        v103 = *(_QWORD *)(a1 + 200);
        v104 = 1;
        v93 = *(_DWORD *)(a1 + 208) + 1;
        v100 = 0;
        v105 = v102 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
        v11 = (__int64 *)(v103 + 16LL * v105);
        v8 = *v11;
        if ( v121 == (__int64 *)*v11 )
          goto LABEL_120;
        while ( v8 != -4096 )
        {
          if ( !v100 && v8 == -8192 )
            v100 = v11;
          v105 = v102 & (v104 + v105);
          v11 = (__int64 *)(v103 + 16LL * v105);
          v8 = *v11;
          if ( v121 == (__int64 *)*v11 )
            goto LABEL_120;
          ++v104;
        }
        goto LABEL_137;
      }
LABEL_124:
      sub_B85070(v108, 2 * v7);
      v94 = *(_DWORD *)(a1 + 216);
      if ( !v94 )
        goto LABEL_164;
      v95 = (__int64)v121;
      v96 = v94 - 1;
      v97 = *(_QWORD *)(a1 + 200);
      v93 = *(_DWORD *)(a1 + 208) + 1;
      v98 = v96 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
      v11 = (__int64 *)(v97 + 16LL * v98);
      v8 = *v11;
      if ( v121 == (__int64 *)*v11 )
        goto LABEL_120;
      v99 = 1;
      v100 = 0;
      while ( v8 != -4096 )
      {
        if ( v8 == -8192 && !v100 )
          v100 = v11;
        v98 = v96 & (v99 + v98);
        v11 = (__int64 *)(v97 + 16LL * v98);
        v8 = *v11;
        if ( v121 == (__int64 *)*v11 )
          goto LABEL_120;
        ++v99;
      }
LABEL_137:
      v8 = v95;
      if ( v100 )
        v11 = v100;
LABEL_120:
      *(_DWORD *)(a1 + 208) = v93;
      if ( *v11 != -4096 )
        --*(_DWORD *)(a1 + 212);
      *v11 = v8;
      v15 = v11 + 1;
      v11[1] = 0;
      goto LABEL_13;
    }
LABEL_6:
    v15 = v13 + 1;
    if ( v13[1] )
    {
      v16 = sub_B85490(v118, v13 + 1);
      v19 = v16;
      if ( *((_BYTE *)v16 + 28) )
      {
        v20 = (__int64 **)v16[1];
        v21 = *((unsigned int *)v16 + 5);
        v22 = &v20[v21];
        v23 = v21;
        if ( v20 != v22 )
        {
          v24 = v20;
          while ( v121 != *v24 )
          {
            if ( v22 == ++v24 )
              goto LABEL_13;
          }
          v25 = (unsigned int)(v23 - 1);
          *((_DWORD *)v19 + 5) = v25;
          *v24 = v20[v25];
          ++*v19;
        }
      }
      else
      {
        v84 = (_QWORD *)sub_C8CA60(v16, v121, v17, v18);
        if ( v84 )
        {
          *v84 = -2;
          ++*((_DWORD *)v19 + 6);
          ++*v19;
        }
      }
    }
LABEL_13:
    *v15 = v120[0];
    v26 = sub_B85490(v118, v120);
    v27 = (__int64)v121;
    v28 = v26;
    if ( !*((_BYTE *)v26 + 28) )
      goto LABEL_21;
    v29 = (__int64 **)v26[1];
    v30 = *((unsigned int *)v28 + 5);
    v31 = &v29[v30];
    if ( v29 == v31 )
    {
LABEL_80:
      if ( (unsigned int)v30 >= *((_DWORD *)v28 + 4) )
      {
LABEL_21:
        sub_C8CC70(v28, v121);
        v32 = v121;
        if ( (__int64 *)v120[0] == v121 )
          goto LABEL_19;
        goto LABEL_22;
      }
      *((_DWORD *)v28 + 5) = v30 + 1;
      *v31 = (__int64 *)v27;
      ++*v28;
      v32 = v121;
    }
    else
    {
      while ( 1 )
      {
        v32 = *v29;
        if ( v121 == *v29 )
          break;
        if ( v31 == ++v29 )
          goto LABEL_80;
      }
    }
    if ( (__int64 *)v120[0] == v32 )
      goto LABEL_19;
LABEL_22:
    v33 = sub_B873F0(a1, v32);
    v34 = *(__int64 **)(v33 + 80);
    v35 = *(unsigned int *)(v33 + 88);
    v122 = v124;
    v36 = &v34[v35];
    v123 = 0xC00000000LL;
    v125 = v127;
    v126 = 0xC00000000LL;
    if ( v34 == v36 )
    {
      v40 = v124;
      v41 = 0;
      goto LABEL_31;
    }
    do
    {
      while ( 1 )
      {
        v37 = sub_B811E0(a1, *v34);
        v38 = *(_QWORD *)(*(_QWORD *)(v37 + 8) + 24LL);
        if ( v109 == *(_DWORD *)(v38 + 384) )
        {
          v60 = (unsigned int)v123;
          if ( (unsigned __int64)(unsigned int)v123 + 1 > HIDWORD(v123) )
          {
            v115 = v37;
            sub_C8D5F0(&v122, v124, (unsigned int)v123 + 1LL, 8);
            v60 = (unsigned int)v123;
            v37 = v115;
          }
          *(_QWORD *)&v122[8 * v60] = v37;
          LODWORD(v123) = v123 + 1;
          goto LABEL_24;
        }
        if ( v109 > *(_DWORD *)(v38 + 384) )
          break;
LABEL_24:
        if ( v36 == ++v34 )
          goto LABEL_30;
      }
      v39 = (unsigned int)v126;
      if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
      {
        v114 = v37;
        sub_C8D5F0(&v125, v127, (unsigned int)v126 + 1LL, 8);
        v39 = (unsigned int)v126;
        v37 = v114;
      }
      ++v34;
      *(_QWORD *)&v125[8 * v39] = v37;
      LODWORD(v126) = v126 + 1;
    }
    while ( v36 != v34 );
LABEL_30:
    v40 = v122;
    v41 = (unsigned int)v123;
LABEL_31:
    sub_B87B60(a1, v40, v41, v120[0]);
    v42 = *(_QWORD *)(v120[0] + 8);
    if ( v42 )
    {
      v43 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v42 + 24) + 16LL))(*(_QWORD *)(v42 + 24));
      sub_B87B60(a1, v125, (unsigned int)v126, v43);
    }
    v44 = sub_B85490(v118, (__int64 *)&v121);
    v45 = (_QWORD *)v44[1];
    if ( *((_BYTE *)v44 + 28) )
      v46 = *((unsigned int *)v44 + 5);
    else
      v46 = *((unsigned int *)v44 + 4);
    v47 = &v45[v46];
    if ( v45 != v47 )
    {
      while ( 1 )
      {
        v48 = *v45;
        v49 = v45;
        if ( *v45 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v47 == ++v45 )
          goto LABEL_38;
      }
      if ( v47 != v45 )
      {
        while ( 1 )
        {
          v66 = *(_DWORD *)(a1 + 216);
          v67 = v120[0];
          if ( !v66 )
            break;
          v68 = *(_QWORD *)(a1 + 200);
          v69 = (v66 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v70 = (_QWORD *)(v68 + 16LL * v69);
          v71 = *v70;
          if ( *v70 == v48 )
            goto LABEL_73;
          v111 = 1;
          v74 = 0;
          while ( 1 )
          {
            if ( v71 == -4096 )
            {
              if ( !v74 )
                v74 = v70;
              v75 = *(_DWORD *)(a1 + 208);
              ++*(_QWORD *)(a1 + 192);
              v76 = v75 + 1;
              if ( 4 * v76 < 3 * v66 )
              {
                if ( v66 - *(_DWORD *)(a1 + 212) - v76 > v66 >> 3 )
                {
LABEL_88:
                  *(_DWORD *)(a1 + 208) = v76;
                  if ( *v74 != -4096 )
                    --*(_DWORD *)(a1 + 212);
                  *v74 = v48;
                  v72 = v74 + 1;
                  v74[1] = 0;
                  goto LABEL_74;
                }
                v107 = v44;
                v113 = v67;
                sub_B85070(v108, v66);
                v85 = *(_DWORD *)(a1 + 216);
                if ( v85 )
                {
                  v86 = v85 - 1;
                  v87 = *(_QWORD *)(a1 + 200);
                  v88 = 0;
                  v89 = v86 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                  v67 = v113;
                  v44 = v107;
                  v90 = 1;
                  v76 = *(_DWORD *)(a1 + 208) + 1;
                  v74 = (_QWORD *)(v87 + 16LL * v89);
                  v91 = *v74;
                  if ( *v74 != v48 )
                  {
                    while ( v91 != -4096 )
                    {
                      if ( v91 == -8192 && !v88 )
                        v88 = v74;
                      v89 = v86 & (v90 + v89);
                      v74 = (_QWORD *)(v87 + 16LL * v89);
                      v91 = *v74;
                      if ( *v74 == v48 )
                        goto LABEL_88;
                      ++v90;
                    }
                    if ( v88 )
                      v74 = v88;
                  }
                  goto LABEL_88;
                }
LABEL_163:
                ++*(_DWORD *)(a1 + 208);
                BUG();
              }
LABEL_95:
              v106 = v44;
              v112 = v67;
              sub_B85070(v108, 2 * v66);
              v77 = *(_DWORD *)(a1 + 216);
              if ( v77 )
              {
                v78 = v77 - 1;
                v79 = *(_QWORD *)(a1 + 200);
                v67 = v112;
                v44 = v106;
                v80 = v78 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                v76 = *(_DWORD *)(a1 + 208) + 1;
                v74 = (_QWORD *)(v79 + 16LL * v80);
                v81 = *v74;
                if ( v48 != *v74 )
                {
                  v82 = 1;
                  v83 = 0;
                  while ( v81 != -4096 )
                  {
                    if ( v81 == -8192 && !v83 )
                      v83 = v74;
                    v80 = v78 & (v82 + v80);
                    v74 = (_QWORD *)(v79 + 16LL * v80);
                    v81 = *v74;
                    if ( *v74 == v48 )
                      goto LABEL_88;
                    ++v82;
                  }
                  if ( v83 )
                    v74 = v83;
                }
                goto LABEL_88;
              }
              goto LABEL_163;
            }
            if ( v74 || v71 != -8192 )
              v70 = v74;
            v69 = (v66 - 1) & (v111 + v69);
            v71 = *(_QWORD *)(v68 + 16LL * v69);
            if ( v71 == v48 )
              break;
            ++v111;
            v74 = v70;
            v70 = (_QWORD *)(v68 + 16LL * v69);
          }
          v70 = (_QWORD *)(v68 + 16LL * v69);
LABEL_73:
          v72 = v70 + 1;
LABEL_74:
          *v72 = v67;
          v73 = v49 + 1;
          if ( v49 + 1 == v47 )
            goto LABEL_38;
          v48 = *v73;
          for ( ++v49; *v73 >= 0xFFFFFFFFFFFFFFFELL; v49 = v73 )
          {
            if ( v47 == ++v73 )
              goto LABEL_38;
            v48 = *v73;
          }
          if ( v49 == v47 )
            goto LABEL_38;
        }
        ++*(_QWORD *)(a1 + 192);
        goto LABEL_95;
      }
    }
LABEL_38:
    v50 = (__int64)v120;
    v110 = v44;
    v51 = sub_B85490(v118, v120);
    v52 = v110;
    v53 = v51;
    v54 = *((_BYTE *)v110 + 28);
    v55 = (__int64 *)v110[1];
    if ( v54 )
    {
      v56 = &v55[*((unsigned int *)v110 + 5)];
      if ( v55 != v56 )
        goto LABEL_40;
      ++*v110;
      goto LABEL_47;
    }
    v56 = &v55[*((unsigned int *)v110 + 4)];
    if ( v55 == v56 )
    {
      ++*v110;
      goto LABEL_43;
    }
LABEL_40:
    while ( 1 )
    {
      v50 = *v55;
      v57 = v55;
      if ( (unsigned __int64)*v55 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v56 == ++v55 )
        goto LABEL_42;
    }
    if ( v55 != v56 )
    {
      v61 = *((_BYTE *)v53 + 28);
      if ( v61 )
      {
LABEL_57:
        v62 = (__int64 *)v53[1];
        v63 = *((unsigned int *)v53 + 5);
        for ( i = &v62[v63]; i != v62; ++v62 )
        {
          if ( *v62 == v50 )
            goto LABEL_61;
        }
        if ( (unsigned int)v63 < *((_DWORD *)v53 + 4) )
        {
          *((_DWORD *)v53 + 5) = v63 + 1;
          *i = v50;
          v61 = *((_BYTE *)v53 + 28);
          ++*v53;
          goto LABEL_61;
        }
      }
      while ( 1 )
      {
        sub_C8CC70(v53, v50);
        v61 = *((_BYTE *)v53 + 28);
LABEL_61:
        v65 = v57 + 1;
        if ( v57 + 1 == v56 )
          break;
        while ( 1 )
        {
          v50 = *v65;
          v57 = v65;
          if ( (unsigned __int64)*v65 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v56 == ++v65 )
            goto LABEL_64;
        }
        if ( v65 == v56 )
          break;
        if ( v61 )
          goto LABEL_57;
      }
LABEL_64:
      v54 = *((_BYTE *)v110 + 28);
      v52 = v110;
    }
LABEL_42:
    ++*v52;
    if ( v54 )
    {
LABEL_47:
      *(_QWORD *)((char *)v52 + 20) = 0;
    }
    else
    {
LABEL_43:
      v58 = 4 * (*((_DWORD *)v52 + 5) - *((_DWORD *)v52 + 6));
      v59 = *((unsigned int *)v52 + 4);
      if ( v58 < 0x20 )
        v58 = 32;
      if ( (unsigned int)v59 <= v58 )
      {
        v50 = 0xFFFFFFFFLL;
        v117 = v52;
        memset((void *)v52[1], -1, 8 * v59);
        v52 = v117;
        goto LABEL_47;
      }
      sub_C8C990(v52);
    }
    if ( v125 != v127 )
      _libc_free(v125, v50);
    if ( v122 != v124 )
      _libc_free(v122, v50);
LABEL_19:
    result = ++v119;
  }
  while ( v116 != v119 );
  return result;
}
