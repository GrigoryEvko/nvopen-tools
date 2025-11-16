// Function: sub_D3EC70
// Address: 0xd3ec70
//
__int64 __fastcall sub_D3EC70(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 v3; // r14
  __int64 v4; // rcx
  __int64 v5; // rax
  int v6; // r13d
  __int64 v7; // r9
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  unsigned int v15; // edx
  __int64 v16; // r12
  unsigned int v17; // edx
  __int64 v18; // rdi
  int v19; // eax
  unsigned int v20; // r15d
  __int64 v21; // rdi
  unsigned int *v22; // rax
  unsigned int *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r10
  signed __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // r12
  __int64 v32; // rbx
  __int64 v33; // rsi
  __int64 *v34; // r13
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rsi
  unsigned __int64 v38; // rax
  __int64 v39; // r10
  unsigned int *v40; // r12
  unsigned int *v41; // rbx
  unsigned int *v42; // rax
  unsigned __int64 v43; // rax
  int v44; // edx
  _BYTE *v45; // rbx
  __int64 v46; // r13
  __int64 v47; // rdi
  __int64 v48; // rbx
  __int64 v49; // r12
  __int64 v50; // rdi
  unsigned int *v51; // r12
  _BOOL4 v52; // r14d
  __int64 v53; // rax
  __int64 v54; // rax
  int *v55; // rdx
  int v56; // ecx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rbx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdi
  int v63; // eax
  __int64 v64; // rdi
  __int64 v65; // rax
  _QWORD *v66; // rbx
  _QWORD *v67; // r12
  _QWORD *v68; // rdi
  int v70; // edx
  unsigned int v71; // r15d
  __int64 v72; // rcx
  __int64 v73; // r11
  __int64 *v74; // rdx
  _QWORD *v75; // rsi
  _BYTE *v76; // rdi
  _BYTE *v77; // rax
  __int64 v78; // rcx
  unsigned int v79; // r10d
  __int64 v80; // rdx
  __int64 v81; // [rsp+0h] [rbp-190h]
  unsigned int v82; // [rsp+24h] [rbp-16Ch]
  __int64 v83; // [rsp+30h] [rbp-160h]
  unsigned int *v84; // [rsp+38h] [rbp-158h]
  _QWORD *v85; // [rsp+38h] [rbp-158h]
  __int64 v86; // [rsp+40h] [rbp-150h]
  unsigned __int64 v87; // [rsp+48h] [rbp-148h]
  __int64 *v88; // [rsp+48h] [rbp-148h]
  __int64 v89; // [rsp+50h] [rbp-140h]
  __int64 v90; // [rsp+58h] [rbp-138h]
  int *v91; // [rsp+60h] [rbp-130h]
  int v92; // [rsp+60h] [rbp-130h]
  unsigned int v93; // [rsp+60h] [rbp-130h]
  __int64 v94; // [rsp+60h] [rbp-130h]
  unsigned int *v95; // [rsp+68h] [rbp-128h]
  unsigned int v96; // [rsp+74h] [rbp-11Ch] BYREF
  unsigned __int64 v97; // [rsp+78h] [rbp-118h] BYREF
  __int64 v98; // [rsp+80h] [rbp-110h] BYREF
  _QWORD *v99; // [rsp+88h] [rbp-108h]
  __int64 v100; // [rsp+90h] [rbp-100h]
  unsigned int v101; // [rsp+98h] [rbp-F8h]
  unsigned int *v102; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v103; // [rsp+A8h] [rbp-E8h]
  char v104[8]; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+B8h] [rbp-D8h] BYREF
  int v106; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v107; // [rsp+C8h] [rbp-C8h]
  int *v108; // [rsp+D0h] [rbp-C0h]
  int *v109; // [rsp+D8h] [rbp-B8h]
  __int64 v110; // [rsp+E0h] [rbp-B0h]
  _BYTE *v111; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+F8h] [rbp-98h]
  _BYTE v113[144]; // [rsp+100h] [rbp-90h] BYREF

  v2 = *(_DWORD *)(a1 + 16);
  v81 = a2;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  if ( !v2 )
  {
    v64 = 0;
    v106 = 0;
    v102 = (unsigned int *)v104;
    v103 = 0x200000000LL;
    v107 = 0;
    v108 = &v106;
    v109 = &v106;
    v110 = 0;
    goto LABEL_105;
  }
  v3 = a1;
  a2 = 0;
  v4 = 0;
  v5 = 0;
  v6 = 0;
  while ( 1 )
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v3 + 8) + 72 * v5 + 16);
    if ( !(_DWORD)a2 )
    {
      ++v98;
      goto LABEL_10;
    }
    v7 = (unsigned int)(a2 - 1);
    v8 = 1;
    v9 = 0;
    LODWORD(v10) = v7 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v11 = v4 + 72LL * (unsigned int)v10;
    v12 = *(_QWORD *)v11;
    if ( v16 != *(_QWORD *)v11 )
    {
      while ( v12 != -4096 )
      {
        if ( !v9 && v12 == -8192 )
          v9 = v11;
        v10 = (unsigned int)v7 & ((_DWORD)v10 + v8);
        v11 = v4 + 72 * v10;
        v12 = *(_QWORD *)v11;
        if ( v16 == *(_QWORD *)v11 )
          goto LABEL_4;
        ++v8;
      }
      if ( v9 )
        v11 = v9;
      ++v98;
      v19 = v100 + 1;
      if ( 4 * ((int)v100 + 1) < (unsigned int)(3 * a2) )
      {
        if ( (int)a2 - (v19 + HIDWORD(v100)) <= (unsigned int)a2 >> 3 )
        {
          sub_D39B00((__int64)&v98, a2);
          if ( !v101 )
          {
LABEL_167:
            LODWORD(v100) = v100 + 1;
            BUG();
          }
          v9 = (__int64)v99;
          v70 = 1;
          a2 = 0;
          v71 = (v101 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v11 = (__int64)&v99[9 * v71];
          v72 = *(_QWORD *)v11;
          v19 = v100 + 1;
          if ( v16 != *(_QWORD *)v11 )
          {
            while ( v72 != -4096 )
            {
              if ( v72 == -8192 && !a2 )
                a2 = v11;
              v7 = (unsigned int)(v70 + 1);
              v80 = (v101 - 1) & (v71 + v70);
              v71 = v80;
              v11 = (__int64)&v99[9 * v80];
              v72 = *(_QWORD *)v11;
              if ( v16 == *(_QWORD *)v11 )
                goto LABEL_12;
              v70 = v7;
            }
            if ( a2 )
              v11 = a2;
          }
        }
        goto LABEL_12;
      }
LABEL_10:
      a2 = (unsigned int)(2 * a2);
      sub_D39B00((__int64)&v98, a2);
      if ( !v101 )
        goto LABEL_167;
      v9 = v101 - 1;
      v7 = (__int64)v99;
      v17 = v9 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v11 = (__int64)&v99[9 * v17];
      v18 = *(_QWORD *)v11;
      v19 = v100 + 1;
      if ( v16 != *(_QWORD *)v11 )
      {
        a2 = 1;
        v78 = 0;
        while ( v18 != -4096 )
        {
          if ( !v78 && v18 == -8192 )
            v78 = v11;
          v79 = a2 + 1;
          v17 = v9 & (a2 + v17);
          a2 = 9LL * v17;
          v11 = (__int64)&v99[9 * v17];
          v18 = *(_QWORD *)v11;
          if ( v16 == *(_QWORD *)v11 )
            goto LABEL_12;
          a2 = v79;
        }
        if ( v78 )
          v11 = v78;
      }
LABEL_12:
      LODWORD(v100) = v19;
      if ( *(_QWORD *)v11 != -4096 )
        --HIDWORD(v100);
      *(_QWORD *)v11 = v16;
      v14 = v11 + 8;
      *(_QWORD *)(v11 + 8) = v11 + 24;
      *(_QWORD *)(v11 + 16) = 0xC00000000LL;
      v13 = 0;
      goto LABEL_6;
    }
LABEL_4:
    v13 = *(unsigned int *)(v11 + 16);
    v14 = v11 + 8;
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 20) )
    {
      a2 = v11 + 24;
      sub_C8D5F0(v11 + 8, (const void *)(v11 + 24), v13 + 1, 4u, v9, v7);
      v13 = *(unsigned int *)(v11 + 16);
    }
LABEL_6:
    *(_DWORD *)(*(_QWORD *)v14 + 4 * v13) = v6;
    v5 = (unsigned int)(v6 + 1);
    ++*(_DWORD *)(v14 + 8);
    v15 = *(_DWORD *)(v3 + 16);
    v6 = v5;
    if ( v15 <= (unsigned int)v5 )
      break;
    v4 = (__int64)v99;
    a2 = v101;
  }
  v106 = 0;
  v102 = (unsigned int *)v104;
  v103 = 0x200000000LL;
  v107 = 0;
  v108 = &v106;
  v109 = &v106;
  v110 = 0;
  if ( v15 )
  {
    v20 = 0;
    v21 = 0;
    v82 = 0;
    while ( 2 )
    {
      v22 = v102;
      v23 = &v102[(unsigned int)v103];
      if ( v102 == v23 )
        goto LABEL_31;
      while ( v82 != *v22 )
      {
        if ( v23 == ++v22 )
          goto LABEL_31;
      }
      if ( v23 == v22 )
        goto LABEL_31;
LABEL_22:
      v21 = ++v82;
      if ( *(_DWORD *)(v3 + 16) > v82 )
      {
        if ( !v110 )
          continue;
        v24 = v107;
        if ( v107 )
        {
          a2 = (__int64)&v106;
          v9 = v82;
          do
          {
            while ( 1 )
            {
              v25 = *(_QWORD *)(v24 + 16);
              v26 = *(_QWORD *)(v24 + 24);
              if ( v82 <= *(_DWORD *)(v24 + 32) )
                break;
              v24 = *(_QWORD *)(v24 + 24);
              if ( !v26 )
                goto LABEL_29;
            }
            a2 = v24;
            v24 = *(_QWORD *)(v24 + 16);
          }
          while ( v25 );
LABEL_29:
          if ( (int *)a2 != &v106 && v82 >= *(_DWORD *)(a2 + 32) )
            goto LABEL_22;
        }
LABEL_31:
        v27 = v81;
        v28 = v81 + 8;
        v29 = (4LL * *(unsigned __int8 *)(*(_QWORD *)(v3 + 8) + 72 * v21 + 40))
            | *(_QWORD *)(*(_QWORD *)(v3 + 8) + 72 * v21 + 16) & 0xFFFFFFFFFFFFFFFBLL;
        v111 = v113;
        v112 = 0x200000000LL;
        v30 = *(_QWORD **)(v81 + 16);
        if ( !v30 )
          goto LABEL_134;
        v31 = v81 + 8;
        v32 = *(_QWORD *)(v81 + 16);
        do
        {
          while ( 1 )
          {
            v33 = *(_QWORD *)(v32 + 16);
            v27 = *(_QWORD *)(v32 + 24);
            if ( v29 <= *(_QWORD *)(v32 + 48) )
              break;
            v32 = *(_QWORD *)(v32 + 24);
            if ( !v27 )
              goto LABEL_36;
          }
          v31 = v32;
          v32 = *(_QWORD *)(v32 + 16);
        }
        while ( v33 );
LABEL_36:
        if ( v31 == v28 )
          goto LABEL_43;
        if ( v29 < *(_QWORD *)(v31 + 48) )
          goto LABEL_43;
        v32 = v31 + 32;
        if ( (*(_BYTE *)(v31 + 40) & 1) != 0 )
          goto LABEL_43;
        v32 = *(_QWORD *)(v31 + 32);
        if ( (*(_BYTE *)(v32 + 8) & 1) != 0 )
          goto LABEL_43;
        v34 = *(__int64 **)v32;
        if ( (*(_BYTE *)(*(_QWORD *)v32 + 8LL) & 1) != 0 )
        {
          v32 = *(_QWORD *)v32;
        }
        else
        {
          v9 = *v34;
          if ( (*(_BYTE *)(*v34 + 8) & 1) == 0 )
          {
            v7 = *(_QWORD *)v9;
            if ( (*(_BYTE *)(*(_QWORD *)v9 + 8LL) & 1) == 0 )
            {
              v73 = *(_QWORD *)v7;
              if ( (*(_BYTE *)(*(_QWORD *)v7 + 8LL) & 1) == 0 )
              {
                v74 = *(__int64 **)v73;
                if ( (*(_BYTE *)(*(_QWORD *)v73 + 8LL) & 1) == 0 )
                {
                  v27 = *v74;
                  if ( (*(_BYTE *)(*v74 + 8) & 1) == 0 )
                  {
                    v75 = *(_QWORD **)v27;
                    if ( (*(_BYTE *)(*(_QWORD *)v27 + 8LL) & 1) == 0 )
                    {
                      v76 = (_BYTE *)*v75;
                      v85 = *(_QWORD **)v27;
                      if ( (*(_BYTE *)(*v75 + 8LL) & 1) == 0 )
                      {
                        v86 = *v74;
                        v88 = *(__int64 **)v73;
                        v89 = *(_QWORD *)v7;
                        v90 = *(_QWORD *)v9;
                        v94 = *v34;
                        v77 = sub_D38E40(v76);
                        v27 = v86;
                        v74 = v88;
                        v73 = v89;
                        v76 = v77;
                        *v85 = v77;
                        v7 = v90;
                        v9 = v94;
                        v28 = v81 + 8;
                      }
                      *(_QWORD *)v27 = v76;
                      v75 = v76;
                    }
                    *v74 = (__int64)v75;
                    v27 = (__int64)v75;
                  }
                  *(_QWORD *)v73 = v27;
                  v74 = (__int64 *)v27;
                }
                *(_QWORD *)v7 = v74;
                v73 = (__int64)v74;
              }
              *(_QWORD *)v9 = v73;
              v7 = v73;
            }
            *v34 = v7;
            v9 = v7;
          }
          *(_QWORD *)v32 = v9;
          v32 = v9;
        }
        *(_QWORD *)(v31 + 32) = v32;
        v30 = *(_QWORD **)(v81 + 16);
        if ( v30 )
        {
LABEL_43:
          v35 = *(_QWORD *)(v32 + 16);
          v36 = v28;
          do
          {
            while ( 1 )
            {
              v37 = v30[2];
              v27 = v30[3];
              if ( v30[6] >= v35 )
                break;
              v30 = (_QWORD *)v30[3];
              if ( !v27 )
                goto LABEL_47;
            }
            v36 = (__int64)v30;
            v30 = (_QWORD *)v30[2];
          }
          while ( v37 );
LABEL_47:
          if ( v28 != v36 && v35 < *(_QWORD *)(v36 + 48) )
            v36 = v28;
        }
        else
        {
LABEL_134:
          v36 = v28;
        }
        if ( (*(_BYTE *)(v36 + 40) & 1) == 0 )
        {
LABEL_73:
          a2 = v3 + 168;
          sub_D38A70((__int64)&v111, (__int64 *)(v3 + 168), v36, v27, v9, v7);
          v48 = (__int64)v111;
          v49 = (__int64)&v111[48 * (unsigned int)v112];
          if ( v111 != (_BYTE *)v49 )
          {
            do
            {
              v49 -= 48;
              v50 = *(_QWORD *)(v49 + 16);
              if ( v50 != v49 + 32 )
                _libc_free(v50, a2);
            }
            while ( v48 != v49 );
            v49 = (__int64)v111;
          }
          if ( (_BYTE *)v49 != v113 )
            _libc_free(v49, a2);
          goto LABEL_22;
        }
        v87 = v36 + 32;
        while ( 2 )
        {
          v38 = *(_QWORD *)(v87 + 16) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v101 )
          {
            v9 = v101 - 1;
            v36 = (unsigned int)v9 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v27 = (__int64)&v99[9 * v36];
            v39 = *(_QWORD *)v27;
            if ( v38 == *(_QWORD *)v27 )
              goto LABEL_54;
            v56 = 1;
            while ( v39 != -4096 )
            {
              v7 = (unsigned int)(v56 + 1);
              v36 = (unsigned int)v9 & (v56 + (_DWORD)v36);
              v27 = (__int64)&v99[9 * (unsigned int)v36];
              v39 = *(_QWORD *)v27;
              if ( v38 == *(_QWORD *)v27 )
                goto LABEL_54;
              v56 = v7;
            }
          }
          v27 = (__int64)&v99[9 * v101];
LABEL_54:
          v40 = *(unsigned int **)(v27 + 8);
          v95 = &v40[*(unsigned int *)(v27 + 16)];
          while ( v95 != v40 )
          {
            v9 = *v40;
            v96 = *v40;
            if ( !v110 )
            {
              v41 = &v102[(unsigned int)v103];
              if ( v102 == v41 )
              {
                if ( (unsigned int)v103 <= 1uLL )
                  goto LABEL_93;
              }
              else
              {
                v42 = v102;
                while ( (_DWORD)v9 != *v42 )
                {
                  if ( v41 == ++v42 )
                    goto LABEL_83;
                }
                if ( v41 != v42 )
                  goto LABEL_61;
LABEL_83:
                if ( (unsigned int)v103 <= 1uLL )
                {
LABEL_93:
                  if ( (unsigned __int64)(unsigned int)v103 + 1 > HIDWORD(v103) )
                  {
                    v93 = v9;
                    sub_C8D5F0((__int64)&v102, v104, (unsigned int)v103 + 1LL, 4u, v9, v7);
                    v9 = v93;
                    v41 = &v102[(unsigned int)v103];
                  }
                  *v41 = v9;
                  LODWORD(v103) = v103 + 1;
                  goto LABEL_61;
                }
                v84 = v40;
                v51 = v102;
                v83 = v3;
                do
                {
                  v54 = sub_B9AB10(&v105, (__int64)&v106, v51);
                  if ( v55 )
                  {
                    v52 = v54 || v55 == &v106 || *v51 < v55[8];
                    v91 = v55;
                    v53 = sub_22077B0(40);
                    *(_DWORD *)(v53 + 32) = *v51;
                    sub_220F040(v52, v53, v91, &v106);
                    ++v110;
                  }
                  ++v51;
                }
                while ( v41 != v51 );
                v40 = v84;
                v3 = v83;
              }
              LODWORD(v103) = 0;
            }
            sub_B99820((__int64)&v105, &v96);
LABEL_61:
            v43 = (unsigned int)v112;
            v44 = v112;
            v45 = &v111[48 * (unsigned int)v112];
            if ( v111 != v45 )
            {
              v46 = (__int64)v111;
              do
              {
                if ( (unsigned int)qword_4F87308 < v20 )
                  break;
                ++v20;
                if ( (unsigned __int8)sub_D345D0(v46, v96, v3) )
                  goto LABEL_71;
                v46 += 48;
              }
              while ( v45 != (_BYTE *)v46 );
              v43 = (unsigned int)v112;
              v44 = v112;
            }
            v27 = HIDWORD(v112);
            if ( v43 >= HIDWORD(v112) )
            {
              v59 = sub_C8D7D0((__int64)&v111, (__int64)v113, 0, 0x30u, &v97, v7);
              v62 = v59 + 48LL * (unsigned int)v112;
              if ( v62 )
                sub_D34480(v62, v96, v3);
              sub_D38990((__int64 *)&v111, v59, v57, v58, v60, v61);
              v63 = v97;
              if ( v111 != v113 )
              {
                v92 = v97;
                _libc_free(v111, v59);
                v63 = v92;
              }
              LODWORD(v112) = v112 + 1;
              v111 = (_BYTE *)v59;
              HIDWORD(v112) = v63;
            }
            else
            {
              v47 = (__int64)&v111[48 * v43];
              if ( v47 )
              {
                sub_D34480(v47, v96, v3);
                v44 = v112;
              }
              v36 = (unsigned int)(v44 + 1);
              LODWORD(v112) = v36;
            }
LABEL_71:
            ++v40;
          }
          v87 = *(_QWORD *)(v87 + 8) & 0xFFFFFFFFFFFFFFFELL;
          if ( !v87 )
            goto LABEL_73;
          continue;
        }
      }
      break;
    }
    v64 = v107;
  }
  else
  {
    v64 = 0;
  }
LABEL_105:
  sub_D32780(v64);
  if ( v102 != (unsigned int *)v104 )
    _libc_free(v102, a2);
  v65 = v101;
  if ( v101 )
  {
    v66 = v99;
    v67 = &v99[9 * v101];
    do
    {
      if ( *v66 != -8192 && *v66 != -4096 )
      {
        v68 = (_QWORD *)v66[1];
        if ( v68 != v66 + 3 )
          _libc_free(v68, a2);
      }
      v66 += 9;
    }
    while ( v67 != v66 );
    v65 = v101;
  }
  return sub_C7D6A0((__int64)v99, 72 * v65, 8);
}
