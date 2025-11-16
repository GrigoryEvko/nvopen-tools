// Function: sub_2CC9AA0
// Address: 0x2cc9aa0
//
char __fastcall sub_2CC9AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r13
  char v9; // di
  char v10; // r15
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  bool v18; // zf
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 *v21; // r13
  __int64 v22; // rsi
  unsigned __int64 v23; // rdi
  unsigned __int64 *v24; // r13
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rdi
  __int64 *v27; // rax
  __int64 *v28; // r13
  __int64 v29; // rsi
  __int64 *v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // r14
  int v33; // r13d
  int v34; // ebx
  __int64 v35; // rsi
  char result; // al
  char v37; // di
  __int64 *v38; // r12
  __int64 *v39; // r14
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 *v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 *v47; // rax
  __int64 *v48; // rax
  unsigned __int64 v49; // rdi
  __int64 v50; // r14
  __int64 v51; // rsi
  __int64 *v52; // rax
  __int64 v53; // r13
  __int64 *v54; // rdx
  __int64 *v55; // rax
  __int64 *v56; // rax
  __int64 *v57; // rax
  __int64 v58; // r12
  __int64 v59; // rsi
  __int64 *v60; // rax
  __int64 *v61; // r12
  __int64 v62; // r14
  int v63; // r15d
  int v64; // r12d
  __int64 v65; // r13
  __int64 v66; // r14
  __int64 v67; // rsi
  __int64 *v68; // rax
  __int64 *v69; // rax
  __int64 *v70; // r14
  __int64 v71; // r15
  __int64 *v72; // r12
  __int64 *v73; // r8
  __int64 v74; // r9
  __int64 v75; // r15
  __int64 *v76; // r14
  int v77; // r12d
  __int64 v78; // rbx
  __int64 v79; // rsi
  __int64 *v80; // rax
  char v81; // [rsp+7h] [rbp-179h]
  __int64 *v82; // [rsp+8h] [rbp-178h]
  __int64 v83; // [rsp+10h] [rbp-170h]
  __int64 *v84; // [rsp+10h] [rbp-170h]
  __int64 *v85; // [rsp+10h] [rbp-170h]
  __int64 v86; // [rsp+18h] [rbp-168h]
  __int64 *v87; // [rsp+20h] [rbp-160h]
  char v89; // [rsp+28h] [rbp-158h]
  char v90; // [rsp+28h] [rbp-158h]
  __int64 v91; // [rsp+38h] [rbp-148h] BYREF
  __int64 v92; // [rsp+40h] [rbp-140h] BYREF
  __int64 *v93; // [rsp+48h] [rbp-138h]
  __int64 v94; // [rsp+50h] [rbp-130h]
  int v95; // [rsp+58h] [rbp-128h]
  char v96; // [rsp+5Ch] [rbp-124h]
  char v97; // [rsp+60h] [rbp-120h] BYREF
  __int64 v98; // [rsp+70h] [rbp-110h] BYREF
  __int64 *v99; // [rsp+78h] [rbp-108h]
  __int64 v100; // [rsp+80h] [rbp-100h]
  int v101; // [rsp+88h] [rbp-F8h]
  char v102; // [rsp+8Ch] [rbp-F4h]
  __int64 v103; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v104; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+A8h] [rbp-D8h]
  __int64 v106; // [rsp+B0h] [rbp-D0h]
  __int64 v107; // [rsp+B8h] [rbp-C8h]
  __int64 v108; // [rsp+C0h] [rbp-C0h]
  unsigned __int64 v109; // [rsp+C8h] [rbp-B8h]
  __int64 v110; // [rsp+D0h] [rbp-B0h]
  __int64 v111; // [rsp+D8h] [rbp-A8h]
  __int64 v112; // [rsp+E0h] [rbp-A0h]
  __int64 *v113; // [rsp+E8h] [rbp-98h]
  __int64 v114; // [rsp+F0h] [rbp-90h] BYREF
  __int64 *v115; // [rsp+F8h] [rbp-88h]
  __int64 v116; // [rsp+100h] [rbp-80h]
  int v117; // [rsp+108h] [rbp-78h]
  char v118; // [rsp+10Ch] [rbp-74h]
  char v119; // [rsp+110h] [rbp-70h] BYREF

  v93 = (__int64 *)&v97;
  v7 = *(_QWORD *)a3;
  v86 = a3;
  v8 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  v92 = 0;
  v94 = 2;
  v95 = 0;
  v96 = 1;
  if ( v8 == v7 )
  {
    v81 = 1;
    goto LABEL_12;
  }
  v9 = 1;
  v10 = 1;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)v7 + 40LL);
      if ( v11 != *(_QWORD *)(a2 + 40) )
        v10 = 0;
      if ( !v9 )
        goto LABEL_41;
      v12 = v93;
      a4 = HIDWORD(v94);
      a3 = (__int64)&v93[HIDWORD(v94)];
      if ( v93 != (__int64 *)a3 )
        break;
LABEL_43:
      if ( HIDWORD(v94) >= (unsigned int)v94 )
      {
LABEL_41:
        v7 += 8;
        sub_C8CC70((__int64)&v92, v11, a3, a4, a5, a6);
        v9 = v96;
        if ( v7 == v8 )
          goto LABEL_11;
      }
      else
      {
        a4 = (unsigned int)(HIDWORD(v94) + 1);
        v7 += 8;
        ++HIDWORD(v94);
        *(_QWORD *)a3 = v11;
        v9 = v96;
        ++v92;
        if ( v7 == v8 )
          goto LABEL_11;
      }
    }
    while ( v11 != *v12 )
    {
      if ( (__int64 *)a3 == ++v12 )
        goto LABEL_43;
    }
    v7 += 8;
  }
  while ( v7 != v8 );
LABEL_11:
  v81 = v10;
LABEL_12:
  v13 = *(_QWORD *)(a2 + 40);
  v114 = 0;
  v115 = (__int64 *)&v119;
  v116 = 8;
  v117 = 0;
  v118 = 1;
  v104 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v105 = 8;
  v104 = sub_22077B0(0x40u);
  v14 = (__int64 *)(((4 * v105 - 4) & 0xFFFFFFFFFFFFFFF8LL) + v104);
  v15 = sub_22077B0(0x200u);
  v18 = v96 == 0;
  v109 = (unsigned __int64)v14;
  v19 = v15 + 512;
  *v14 = v15;
  v107 = v15;
  v111 = v15;
  v106 = v15;
  v110 = v15;
  v99 = &v103;
  v100 = 0x100000002LL;
  v20 = v93;
  v108 = v19;
  v113 = v14;
  v112 = v19;
  v101 = 0;
  v102 = 1;
  v103 = v13;
  v98 = 1;
  if ( v18 )
    v21 = &v93[(unsigned int)v94];
  else
    v21 = &v93[HIDWORD(v94)];
  if ( v93 != v21 )
  {
    while ( 1 )
    {
      v22 = *v20;
      if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v21 == ++v20 )
        goto LABEL_17;
    }
    if ( v21 != v20 )
    {
      v37 = 1;
      v83 = a2;
      v38 = v20;
      v39 = v21;
LABEL_48:
      v40 = v99;
      v41 = HIDWORD(v100);
      v42 = &v99[HIDWORD(v100)];
      if ( v99 != v42 )
      {
        while ( *v40 != v22 )
        {
          if ( v42 == ++v40 )
            goto LABEL_107;
        }
LABEL_52:
        if ( v13 != v22 )
        {
LABEL_94:
          v53 = *(_QWORD *)(v22 + 16);
          if ( v53 )
          {
            while ( 1 )
            {
              v41 = *(_QWORD *)(v53 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v41 - 30) <= 0xAu )
                break;
              v53 = *(_QWORD *)(v53 + 8);
              if ( !v53 )
                goto LABEL_53;
            }
LABEL_98:
            if ( v13 != *(_QWORD *)(v41 + 40) )
            {
              v91 = *(_QWORD *)(v41 + 40);
              sub_2CC8EF0((unsigned __int64 *)&v104, &v91);
            }
            while ( 1 )
            {
              v53 = *(_QWORD *)(v53 + 8);
              if ( !v53 )
                break;
              v41 = *(_QWORD *)(v53 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v41 - 30) <= 0xAu )
                goto LABEL_98;
            }
            v37 = v102;
          }
        }
        goto LABEL_53;
      }
LABEL_107:
      if ( HIDWORD(v100) < (unsigned int)v100 )
      {
        v41 = (unsigned int)++HIDWORD(v100);
        *v42 = v22;
        v37 = v102;
        ++v98;
        v22 = *v38;
        goto LABEL_52;
      }
      while ( 1 )
      {
        sub_C8CC70((__int64)&v98, v22, (__int64)v42, v41, v16, v17);
        v22 = *v38;
        v37 = v102;
        if ( v13 != *v38 )
          goto LABEL_94;
LABEL_53:
        v43 = v38 + 1;
        if ( v38 + 1 == v39 )
          break;
        while ( 1 )
        {
          v22 = *v43;
          v38 = v43;
          if ( (unsigned __int64)*v43 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v39 == ++v43 )
            goto LABEL_56;
        }
        if ( v39 == v43 )
          break;
        if ( v37 )
          goto LABEL_48;
      }
LABEL_56:
      v44 = v106;
      v45 = v110;
      a2 = v83;
      if ( v106 != v110 )
      {
LABEL_57:
        while ( 2 )
        {
          if ( v111 == v45 )
            v45 = *(v113 - 1) + 512;
          v46 = *(_QWORD *)(v45 - 8);
          if ( v37 )
          {
            v47 = v99;
            v41 = HIDWORD(v100);
            v44 = (__int64)&v99[HIDWORD(v100)];
            if ( v99 != (__int64 *)v44 )
            {
              while ( v46 != *v47 )
              {
                if ( (__int64 *)v44 == ++v47 )
                  goto LABEL_105;
              }
LABEL_64:
              if ( v118 )
              {
                v48 = v115;
                v41 = HIDWORD(v116);
                v44 = (__int64)&v115[HIDWORD(v116)];
                if ( v115 != (__int64 *)v44 )
                {
                  while ( v46 != *v48 )
                  {
                    if ( (__int64 *)v44 == ++v48 )
                      goto LABEL_109;
                  }
LABEL_69:
                  v49 = v110;
                  if ( v110 != v111 )
                  {
LABEL_70:
                    v110 = v49 - 8;
                    goto LABEL_71;
                  }
LABEL_103:
                  j_j___libc_free_0(v49);
                  v44 = *--v113 + 512;
                  v111 = *v113;
                  v112 = v44;
                  v110 = v111 + 504;
LABEL_71:
                  v50 = *(_QWORD *)(v46 + 16);
                  v37 = v102;
                  if ( v50 )
                  {
                    while ( 1 )
                    {
                      v44 = *(_QWORD *)(v50 + 24);
                      if ( (unsigned __int8)(*(_BYTE *)v44 - 30) <= 0xAu )
                        break;
                      v50 = *(_QWORD *)(v50 + 8);
                      if ( !v50 )
                      {
                        v45 = v110;
                        if ( v110 == v106 )
                          goto LABEL_86;
                        goto LABEL_57;
                      }
                    }
                    v51 = *(_QWORD *)(v44 + 40);
                    if ( v102 )
                    {
LABEL_74:
                      v52 = v99;
                      v44 = (__int64)&v99[HIDWORD(v100)];
                      if ( v99 == (__int64 *)v44 )
                        goto LABEL_83;
                      while ( v51 != *v52 )
                      {
                        if ( (__int64 *)v44 == ++v52 )
                          goto LABEL_83;
                      }
                      goto LABEL_78;
                    }
                    while ( 1 )
                    {
                      if ( !sub_C8CA60((__int64)&v98, v51) )
                      {
                        v51 = *(_QWORD *)(*(_QWORD *)(v50 + 24) + 40LL);
LABEL_83:
                        v91 = v51;
                        sub_2CC8EF0((unsigned __int64 *)&v104, &v91);
                      }
                      v50 = *(_QWORD *)(v50 + 8);
                      v37 = v102;
                      if ( !v50 )
                        break;
                      while ( 1 )
                      {
                        v44 = *(_QWORD *)(v50 + 24);
                        if ( (unsigned __int8)(*(_BYTE *)v44 - 30) <= 0xAu )
                          break;
LABEL_78:
                        v50 = *(_QWORD *)(v50 + 8);
                        if ( !v50 )
                          goto LABEL_85;
                      }
                      v51 = *(_QWORD *)(v44 + 40);
                      if ( v37 )
                        goto LABEL_74;
                    }
                  }
LABEL_85:
                  v45 = v110;
                  if ( v110 == v106 )
                    goto LABEL_86;
                  continue;
                }
LABEL_109:
                if ( HIDWORD(v116) < (unsigned int)v116 )
                {
                  v41 = (unsigned int)++HIDWORD(v116);
                  *(_QWORD *)v44 = v46;
                  ++v114;
                  goto LABEL_69;
                }
              }
              sub_C8CC70((__int64)&v114, v46, v44, v41, v16, v17);
              v49 = v110;
              if ( v110 != v111 )
                goto LABEL_70;
              goto LABEL_103;
            }
LABEL_105:
            if ( HIDWORD(v100) < (unsigned int)v100 )
            {
              v41 = (unsigned int)++HIDWORD(v100);
              *(_QWORD *)v44 = v46;
              ++v98;
              goto LABEL_64;
            }
          }
          break;
        }
        sub_C8CC70((__int64)&v98, v46, v44, v41, v16, v17);
        goto LABEL_64;
      }
LABEL_86:
      if ( !v37 )
        _libc_free((unsigned __int64)v99);
    }
  }
LABEL_17:
  v23 = v104;
  if ( v104 )
  {
    v24 = (unsigned __int64 *)v109;
    v25 = (unsigned __int64 *)(v113 + 1);
    if ( (unsigned __int64)(v113 + 1) > v109 )
    {
      do
      {
        v26 = *v24++;
        j_j___libc_free_0(v26);
      }
      while ( v25 > v24 );
      v23 = v104;
    }
    j_j___libc_free_0(v23);
  }
  v27 = v115;
  if ( v118 )
    v28 = &v115[HIDWORD(v116)];
  else
    v28 = &v115[(unsigned int)v116];
  if ( v115 != v28 )
  {
    while ( 1 )
    {
      v29 = *v27;
      v30 = v27;
      if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v28 == ++v27 )
        goto LABEL_27;
    }
    if ( v28 != v27 )
    {
      if ( !v96 )
        goto LABEL_127;
LABEL_116:
      v54 = &v93[HIDWORD(v94)];
      v55 = v93;
      if ( v93 != v54 )
      {
        while ( *v55 != v29 )
        {
          if ( v54 == ++v55 )
            goto LABEL_121;
        }
        --HIDWORD(v94);
        *v55 = v93[HIDWORD(v94)];
        ++v92;
      }
LABEL_121:
      while ( 1 )
      {
        v56 = v30 + 1;
        if ( v30 + 1 == v28 )
          break;
        v29 = *v56;
        for ( ++v30; (unsigned __int64)*v56 >= 0xFFFFFFFFFFFFFFFELL; v30 = v56 )
        {
          if ( v28 == ++v56 )
            goto LABEL_27;
          v29 = *v56;
        }
        if ( v30 == v28 )
          break;
        if ( v96 )
          goto LABEL_116;
LABEL_127:
        v57 = sub_C8CA60((__int64)&v92, v29);
        if ( v57 )
        {
          *v57 = -2;
          ++v95;
          ++v92;
        }
      }
    }
  }
LABEL_27:
  v31 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 <= 1 )
    v31 = **(_QWORD **)(v31 + 16);
  v32 = a2 + 24;
  v33 = *(_DWORD *)(v31 + 8) >> 8;
  v34 = *(_DWORD *)(v86 + 8);
  if ( v81 )
  {
    if ( v34 > 0 )
    {
      while ( 1 )
      {
        v32 = *(_QWORD *)(v32 + 8);
        LOBYTE(v104) = 0;
        v35 = v32 - 24;
        if ( !v32 )
          v35 = 0;
        if ( sub_2CC97E0(a1, v35, v33, v86, &v104) )
          break;
        if ( (_BYTE)v104 )
        {
          if ( !--v34 )
            goto LABEL_36;
        }
      }
LABEL_169:
      result = !((unsigned __int8)v104 & (v34 == 1));
      goto LABEL_37;
    }
LABEL_36:
    result = 0;
    goto LABEL_37;
  }
  v58 = *(_QWORD *)(a2 + 40) + 48LL;
  if ( v58 == v32 )
  {
LABEL_134:
    v60 = v115;
    if ( v118 )
      v61 = &v115[HIDWORD(v116)];
    else
      v61 = &v115[(unsigned int)v116];
    if ( v115 != v61 )
    {
      v62 = *v115;
      if ( (unsigned __int64)*v115 < 0xFFFFFFFFFFFFFFFELL )
      {
LABEL_140:
        v84 = v60;
        if ( v60 != v61 )
        {
          v82 = v61;
          v63 = v34;
          v64 = v33;
          while ( 1 )
          {
            v65 = *(_QWORD *)(v62 + 56);
            v66 = v62 + 48;
            if ( v66 != v65 )
              break;
LABEL_147:
            v68 = v84 + 1;
            if ( v84 + 1 != v82 )
            {
              while ( 1 )
              {
                v62 = *v68;
                if ( (unsigned __int64)*v68 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v82 == ++v68 )
                  goto LABEL_150;
              }
              v84 = v68;
              if ( v68 != v82 )
                continue;
            }
LABEL_150:
            v34 = v63;
            v33 = v64;
            goto LABEL_151;
          }
          while ( 1 )
          {
            v67 = v65 - 24;
            if ( !v65 )
              v67 = 0;
            LOBYTE(v104) = 0;
            result = sub_2CC97E0(a1, v67, v64, v86, &v104);
            if ( result )
              goto LABEL_37;
            v65 = *(_QWORD *)(v65 + 8);
            v63 = ((_BYTE)v104 == 0) + v63 - 1;
            if ( v66 == v65 )
              goto LABEL_147;
          }
        }
      }
      else
      {
        while ( v61 != ++v60 )
        {
          v62 = *v60;
          if ( (unsigned __int64)*v60 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_140;
        }
      }
    }
LABEL_151:
    v69 = v93;
    if ( v96 )
      v70 = &v93[HIDWORD(v94)];
    else
      v70 = &v93[(unsigned int)v94];
    if ( v93 != v70 )
    {
      while ( 1 )
      {
        v71 = *v69;
        v72 = v69;
        if ( (unsigned __int64)*v69 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v70 == ++v69 )
          goto LABEL_36;
      }
      if ( v69 != v70 )
      {
        v73 = &v104;
        do
        {
          v74 = *(_QWORD *)(v71 + 56);
          v75 = v71 + 48;
          if ( v75 != v74 )
          {
            v87 = v70;
            v76 = v72;
            v77 = v34;
            v78 = v74;
            do
            {
              v79 = v78 - 24;
              if ( !v78 )
                v79 = 0;
              v85 = v73;
              LOBYTE(v104) = 0;
              result = sub_2CC97E0(a1, v79, v33, v86, v73);
              v73 = v85;
              if ( result )
              {
                v34 = v77;
                goto LABEL_169;
              }
              v77 = ((_BYTE)v104 == 0) + v77 - 1;
              if ( !v77 )
                goto LABEL_37;
              v78 = *(_QWORD *)(v78 + 8);
            }
            while ( v75 != v78 );
            v34 = v77;
            v72 = v76;
            v70 = v87;
          }
          v80 = v72 + 1;
          if ( v72 + 1 == v70 )
            break;
          while ( 1 )
          {
            v71 = *v80;
            v72 = v80;
            if ( (unsigned __int64)*v80 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v70 == ++v80 )
              goto LABEL_36;
          }
        }
        while ( v80 != v70 );
      }
    }
    goto LABEL_36;
  }
  while ( 1 )
  {
    v59 = v32 - 24;
    if ( !v32 )
      v59 = 0;
    LOBYTE(v104) = 0;
    result = sub_2CC97E0(a1, v59, v33, v86, &v104);
    if ( result )
      break;
    v32 = *(_QWORD *)(v32 + 8);
    v34 = ((_BYTE)v104 == 0) + v34 - 1;
    if ( v58 == v32 )
      goto LABEL_134;
  }
LABEL_37:
  if ( !v118 )
  {
    v90 = result;
    _libc_free((unsigned __int64)v115);
    result = v90;
  }
  if ( !v96 )
  {
    v89 = result;
    _libc_free((unsigned __int64)v93);
    return v89;
  }
  return result;
}
