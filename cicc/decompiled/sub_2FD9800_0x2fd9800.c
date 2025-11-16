// Function: sub_2FD9800
// Address: 0x2fd9800
//
__int64 __fastcall sub_2FD9800(
        __int64 *a1,
        char a2,
        __int64 *a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 *v7; // r15
  __int64 v8; // r14
  bool v9; // al
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned int v13; // r13d
  unsigned int v15; // ebx
  int v16; // r14d
  unsigned int v17; // ecx
  int *v18; // rdx
  int v19; // eax
  int v20; // r12d
  unsigned int v21; // eax
  int v22; // edx
  int v23; // ecx
  __int64 v24; // rdi
  int v25; // r10d
  __int64 *v26; // r13
  __int64 v27; // r12
  __int64 *v28; // rsi
  unsigned int v29; // ebx
  __int64 v30; // r12
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 *v38; // rbx
  __int64 *v39; // rcx
  __int64 *v40; // r13
  __int64 v41; // r15
  int v42; // eax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rsi
  __int64 *v47; // rbx
  __int64 v48; // r13
  unsigned int *p_i; // r12
  _QWORD *v50; // rdi
  __int64 v51; // r15
  _QWORD *v52; // rsi
  int v53; // esi
  unsigned int v54; // r15d
  int v55; // eax
  __int64 *v56; // r13
  __int64 *v57; // r12
  __int64 *v58; // rsi
  __int64 v59; // rax
  _BYTE *v60; // rsi
  unsigned int *v61; // rbx
  _BYTE *v62; // rax
  _BYTE *v63; // r12
  __int64 (*v64)(); // rax
  unsigned __int64 v65; // rbx
  int v66; // r12d
  unsigned __int64 *v67; // r12
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // r13
  __int64 v79; // rsi
  bool v80; // [rsp+10h] [rbp-240h]
  __int64 *v82; // [rsp+20h] [rbp-230h]
  __int64 *v83; // [rsp+20h] [rbp-230h]
  __int64 v84; // [rsp+28h] [rbp-228h]
  __int64 *v86; // [rsp+30h] [rbp-220h]
  unsigned __int8 v89; // [rsp+40h] [rbp-210h]
  __int64 v90; // [rsp+40h] [rbp-210h]
  __int64 *v92; // [rsp+48h] [rbp-208h]
  __int64 v93; // [rsp+48h] [rbp-208h]
  __int64 *v94; // [rsp+50h] [rbp-200h]
  bool v95; // [rsp+50h] [rbp-200h]
  __int64 v96; // [rsp+58h] [rbp-1F8h]
  __int64 *v97; // [rsp+58h] [rbp-1F8h]
  __int64 v98; // [rsp+68h] [rbp-1E8h] BYREF
  __int64 v99; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v100; // [rsp+78h] [rbp-1D8h] BYREF
  __int64 v101; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v102; // [rsp+88h] [rbp-1C8h]
  __int64 v103; // [rsp+90h] [rbp-1C0h]
  __int64 v104; // [rsp+98h] [rbp-1B8h]
  __int64 v105; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v106; // [rsp+A8h] [rbp-1A8h]
  __int64 v107; // [rsp+B0h] [rbp-1A0h]
  unsigned int v108; // [rsp+B8h] [rbp-198h]
  __int64 *i; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v110; // [rsp+C8h] [rbp-188h]
  __int64 v111; // [rsp+D0h] [rbp-180h] BYREF
  unsigned int v112; // [rsp+D8h] [rbp-178h]
  __int64 v113; // [rsp+100h] [rbp-150h] BYREF
  __int64 v114; // [rsp+108h] [rbp-148h]
  __int64 v115; // [rsp+110h] [rbp-140h]
  __int64 v116; // [rsp+118h] [rbp-138h]
  __int64 *v117; // [rsp+120h] [rbp-130h]
  __int64 v118; // [rsp+128h] [rbp-128h]
  _BYTE v119[64]; // [rsp+130h] [rbp-120h] BYREF
  _BYTE *v120; // [rsp+170h] [rbp-E0h] BYREF
  __int64 v121; // [rsp+178h] [rbp-D8h]
  _BYTE v122[208]; // [rsp+180h] [rbp-D0h] BYREF

  v7 = a1;
  v8 = (__int64)a3;
  v9 = sub_2E32580(a3);
  v12 = *(_QWORD *)(v8 + 56);
  v80 = v9;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v96 = v8 + 48;
  if ( v12 != v8 + 48 )
  {
    v84 = v8;
    while ( 1 )
    {
      if ( *(_WORD *)(v12 + 68) != 68 && *(_WORD *)(v12 + 68) )
      {
LABEL_5:
        v7 = a1;
        v8 = v84;
        goto LABEL_6;
      }
      v15 = 1;
      v16 = *(_DWORD *)(v12 + 40) & 0xFFFFFF;
      if ( v16 != 1 )
        break;
LABEL_22:
      if ( (*(_BYTE *)v12 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
          v12 = *(_QWORD *)(v12 + 8);
      }
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v96 )
        goto LABEL_5;
    }
    while ( 1 )
    {
      v20 = *(_DWORD *)(*(_QWORD *)(v12 + 32) + 40LL * v15 + 8);
      if ( !(_DWORD)v104 )
        break;
      v10 = (unsigned int)(v104 - 1);
      v17 = v10 & (37 * v20);
      v18 = (int *)(v102 + 4LL * v17);
      v19 = *v18;
      if ( v20 != *v18 )
      {
        v25 = 1;
        v11 = 0;
        while ( v19 != -1 )
        {
          if ( v19 != -2 || v11 )
            v18 = (int *)v11;
          v11 = (unsigned int)(v25 + 1);
          v17 = v10 & (v25 + v17);
          v19 = *(_DWORD *)(v102 + 4LL * v17);
          if ( v20 == v19 )
            goto LABEL_12;
          ++v25;
          v11 = (__int64)v18;
          v18 = (int *)(v102 + 4LL * v17);
        }
        if ( !v11 )
          v11 = (__int64)v18;
        ++v101;
        v22 = v103 + 1;
        if ( 4 * ((int)v103 + 1) < (unsigned int)(3 * v104) )
        {
          if ( (int)v104 - HIDWORD(v103) - v22 <= (unsigned int)v104 >> 3 )
          {
            sub_2E29BA0((__int64)&v101, v104);
            if ( !(_DWORD)v104 )
            {
LABEL_170:
              LODWORD(v103) = v103 + 1;
              BUG();
            }
            v53 = 1;
            v10 = 0;
            v54 = (v104 - 1) & (37 * v20);
            v11 = v102 + 4LL * v54;
            v22 = v103 + 1;
            v55 = *(_DWORD *)v11;
            if ( v20 != *(_DWORD *)v11 )
            {
              while ( v55 != -1 )
              {
                if ( !v10 && v55 == -2 )
                  v10 = v11;
                v54 = (v104 - 1) & (v53 + v54);
                v11 = v102 + 4LL * v54;
                v55 = *(_DWORD *)v11;
                if ( v20 == *(_DWORD *)v11 )
                  goto LABEL_34;
                ++v53;
              }
              if ( v10 )
                v11 = v10;
            }
          }
          goto LABEL_34;
        }
LABEL_15:
        sub_2E29BA0((__int64)&v101, 2 * v104);
        if ( !(_DWORD)v104 )
          goto LABEL_170;
        v21 = (v104 - 1) & (37 * v20);
        v11 = v102 + 4LL * v21;
        v22 = v103 + 1;
        v23 = *(_DWORD *)v11;
        if ( v20 != *(_DWORD *)v11 )
        {
          v10 = 1;
          v24 = 0;
          while ( v23 != -1 )
          {
            if ( !v24 && v23 == -2 )
              v24 = v11;
            v21 = (v104 - 1) & (v10 + v21);
            v11 = v102 + 4LL * v21;
            v23 = *(_DWORD *)v11;
            if ( v20 == *(_DWORD *)v11 )
              goto LABEL_34;
            v10 = (unsigned int)(v10 + 1);
          }
          if ( v24 )
            v11 = v24;
        }
LABEL_34:
        LODWORD(v103) = v22;
        if ( *(_DWORD *)v11 != -1 )
          --HIDWORD(v103);
        *(_DWORD *)v11 = v20;
      }
LABEL_12:
      v15 += 2;
      if ( v16 == v15 )
        goto LABEL_22;
    }
    ++v101;
    goto LABEL_15;
  }
LABEL_6:
  if ( a2 )
  {
    v13 = sub_2FD6CE0(v7, v8, (__int64 *)a5, (__int64)&v101, v10, v11);
    goto LABEL_8;
  }
  v113 = 0;
  v117 = (__int64 *)v119;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v118 = 0x800000000LL;
  if ( a7 )
  {
    v26 = *(__int64 **)a7;
    v27 = *(_QWORD *)a7 + 8LL * *(unsigned int *)(a7 + 8);
    if ( v27 == *(_QWORD *)a7 )
      goto LABEL_123;
    do
    {
      v28 = v26++;
      sub_2FD92E0((__int64)&v113, v28);
    }
    while ( (__int64 *)v27 != v26 );
  }
  else
  {
    v56 = *(__int64 **)(v8 + 64);
    v57 = &v56[*(unsigned int *)(v8 + 72)];
    if ( v57 == v56 )
      goto LABEL_123;
    do
    {
      v58 = v56++;
      sub_2FD92E0((__int64)&v113, v58);
    }
    while ( v57 != v56 );
  }
  v82 = &v117[(unsigned int)v118];
  if ( v117 != v82 )
  {
    v94 = v117;
    v29 = 0;
    while ( 1 )
    {
      v30 = *v94;
      v89 = sub_2FD7360(v7, v8, *v94);
      if ( v89 )
      {
        v33 = (__int64 *)v7[4];
        if ( *(_DWORD *)(v33[1] + 544) == 21 || (sub_B2EE70((__int64)&v120, *v33, 0), v122[0]) && *((_BYTE *)v7 + 57) )
        {
LABEL_49:
          v34 = *(unsigned int *)(a5 + 8);
          if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            sub_C8D5F0(a5, (const void *)(a5 + 16), v34 + 1, 8u, v31, v32);
            v34 = *(unsigned int *)(a5 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a5 + 8 * v34) = v30;
          ++*(_DWORD *)(a5 + 8);
          (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)*v7 + 360LL))(*v7, v30, 0);
          v35 = *(_QWORD *)(v8 + 56);
          i = 0;
          v110 = 0;
          v120 = v122;
          v111 = 0;
          v112 = 0;
          v121 = 0x400000000LL;
          if ( v35 != v96 )
          {
            while ( 1 )
            {
              if ( !v35 )
                BUG();
              v36 = v35;
              if ( (*(_BYTE *)v35 & 4) == 0 && (*(_BYTE *)(v35 + 44) & 8) != 0 )
              {
                do
                  v36 = *(_QWORD *)(v36 + 8);
                while ( (*(_BYTE *)(v36 + 44) & 8) != 0 );
              }
              v37 = *(_QWORD *)(v36 + 8);
              if ( *(_WORD *)(v35 + 68) == 68 || !*(_WORD *)(v35 + 68) )
              {
                sub_2FD8650(v7, v35, v8, v30, (__int64)&i, (unsigned int *)&v120, (__int64)&v101, 1);
                if ( v37 == v96 )
                  break;
              }
              else
              {
                sub_2FD88F0((__int64)v7, v35, v8, (_QWORD *)v30, (__int64)&i, (__int64)&v101);
                if ( v37 == v96 )
                  break;
              }
              v35 = v37;
            }
          }
          v38 = v7;
          sub_2FD7440((__int64)v7, v30, (__int64)&v120, a6);
          sub_2E33590(v30, *(__int64 **)(v30 + 112), 0);
          v39 = *(__int64 **)(v8 + 112);
          v40 = v39;
          v92 = &v39[*(unsigned int *)(v8 + 120)];
          if ( v39 != v92 )
          {
            do
            {
              v41 = *v40++;
              v42 = sub_2E441D0(v38[2], v8, v41);
              sub_2E33F80(v30, v41, v42, v43, v44, v45);
            }
            while ( v92 != v40 );
            v7 = v38;
          }
          if ( v80 )
          {
            v46 = *(_QWORD *)(v8 + 8);
            if ( v46 == *(_QWORD *)(v8 + 32) + 320LL )
              v46 = 0;
            sub_2E32A60(v30, v46);
          }
          if ( v120 != v122 )
            _libc_free((unsigned __int64)v120);
          sub_C7D6A0(v110, 12LL * v112, 4);
          v29 = v89;
          goto LABEL_42;
        }
        if ( a4 )
        {
          if ( v30 != a4 )
            goto LABEL_49;
        }
        else if ( !sub_2E322F0(v30, v8) || !sub_2E32580((__int64 *)v30) )
        {
          goto LABEL_49;
        }
      }
LABEL_42:
      if ( v82 == ++v94 )
      {
        v13 = v29;
        goto LABEL_72;
      }
    }
  }
LABEL_123:
  v13 = 0;
LABEL_72:
  if ( !a4 )
    a4 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
  v98 = 0;
  v99 = 0;
  v120 = v122;
  v121 = 0x400000000LL;
  if ( *(_DWORD *)(a4 + 120) == 1 && **(_QWORD **)(a4 + 112) == v8 )
  {
    v64 = *(__int64 (**)())(*(_QWORD *)*v7 + 344LL);
    if ( v64 != sub_2DB1AE0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v64)(
             *v7,
             a4,
             &v98,
             &v99,
             &v120,
             0)
        || (_DWORD)v121
        || !(v95 = v98 == 0 || v98 == v8)
        || *(_DWORD *)(v8 + 72) != 1
        || *(_BYTE *)(v8 + 217)
        || *(_QWORD *)(v8 + 224) )
      {
        if ( *((_BYTE *)v7 + 56) && (_BYTE)v13 )
          goto LABEL_77;
      }
      else
      {
        v65 = a4 + 48;
        v66 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)*v7 + 360LL))(*v7, a4, 0);
        if ( a4 + 48 == sub_2E313E0(a4) )
        {
          if ( *((_BYTE *)v7 + 56) )
          {
            v105 = 0;
            v106 = 0;
            v110 = 0x400000000LL;
            v77 = *(_QWORD *)(v8 + 56);
            v107 = 0;
            v108 = 0;
            v100 = v77;
            for ( i = &v111; ; sub_2FD8650(v7, v78, v8, a4, (__int64)&v105, (unsigned int *)&i, (__int64)&v101, 1) )
            {
              v78 = v100;
              if ( v100 == v96 || *(_WORD *)(v100 + 68) && *(_WORD *)(v100 + 68) != 68 )
                break;
              sub_2FD79B0(&v100);
            }
            while ( v78 != v96 )
            {
              sub_2FD79B0(&v100);
              sub_2FD88F0((__int64)v7, v78, v8, (_QWORD *)a4, (__int64)&v105, (__int64)&v101);
              sub_2E88E20(v78);
              v78 = v100;
            }
            sub_2FD7440((__int64)v7, a4, (__int64)&i, a6);
            if ( i != &v111 )
              _libc_free((unsigned __int64)i);
            sub_C7D6A0(v106, 12LL * v108, 4);
          }
          else
          {
            (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)*v7 + 360LL))(*v7, a4, 0);
            v67 = *(unsigned __int64 **)(v8 + 56);
            if ( v67 != (unsigned __int64 *)v96 && v65 != v96 )
            {
              sub_2E310C0((__int64 *)(a4 + 40), (__int64 *)(v8 + 40), (__int64)v67, v96);
              if ( (unsigned __int64 *)v96 != v67 )
              {
                v68 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)((*v67 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v96;
                *(_QWORD *)(v8 + 48) = *(_QWORD *)(v8 + 48) & 7LL | *v67 & 0xFFFFFFFFFFFFFFF8LL;
                v69 = *(_QWORD *)(a4 + 48);
                *(_QWORD *)(v68 + 8) = v65;
                *v67 = v69 & 0xFFFFFFFFFFFFFFF8LL | *v67 & 7;
                *(_QWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v67;
                *(_QWORD *)(a4 + 48) = v68 | *(_QWORD *)(a4 + 48) & 7LL;
              }
            }
          }
          sub_2E33590(a4, *(__int64 **)(a4 + 112), 0);
          sub_2E340B0(a4, v8, v70, v71, v72, v73);
          if ( v80 )
          {
            v79 = *(_QWORD *)(v8 + 8);
            if ( v79 == *(_QWORD *)(v8 + 32) + 320LL )
              v79 = 0;
            sub_2E32A60(a4, v79);
          }
          v76 = *(unsigned int *)(a5 + 8);
          if ( v76 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            sub_C8D5F0(a5, (const void *)(a5 + 16), v76 + 1, 8u, v74, v75);
            v76 = *(unsigned int *)(a5 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a5 + 8 * v76) = a4;
          ++*(_DWORD *)(a5 + 8);
LABEL_132:
          if ( !*((_BYTE *)v7 + 56) )
          {
            v13 = v95;
            goto LABEL_83;
          }
LABEL_77:
          v47 = v117;
          v97 = &v117[(unsigned int)v118];
          if ( v97 != v117 )
          {
            v83 = v7;
            v48 = a5;
            p_i = (unsigned int *)&i;
            do
            {
              v50 = *(_QWORD **)v48;
              v51 = *v47;
              v52 = (_QWORD *)(*(_QWORD *)v48 + 8LL * *(unsigned int *)(v48 + 8));
              v100 = *v47;
              if ( v52 == sub_2FD5C20(v50, (__int64)v52, &v100) && *(_DWORD *)(v51 + 120) == 1 )
              {
                v105 = 0;
                v106 = 0;
                i = &v111;
                v107 = 0;
                v108 = 0;
                v110 = 0x400000000LL;
                v59 = sub_2E311E0(v8);
                v60 = *(_BYTE **)(v8 + 56);
                v90 = v59;
                if ( v60 == (_BYTE *)v59 )
                {
                  sub_2FD7440((__int64)v83, v51, (__int64)p_i, a6);
                }
                else
                {
                  v86 = v47;
                  v61 = p_i;
                  while ( 1 )
                  {
                    if ( !v60 )
                      BUG();
                    v62 = v60;
                    if ( (*v60 & 4) == 0 && (v60[44] & 8) != 0 )
                    {
                      do
                        v62 = (_BYTE *)*((_QWORD *)v62 + 1);
                      while ( (v62[44] & 8) != 0 );
                    }
                    v63 = (_BYTE *)*((_QWORD *)v62 + 1);
                    v93 = (__int64)v61;
                    sub_2FD8650(v83, (__int64)v60, v8, v51, (__int64)&v105, v61, (__int64)&v101, 0);
                    if ( (_BYTE *)v90 == v63 )
                      break;
                    v60 = v63;
                  }
                  p_i = v61;
                  v47 = v86;
                  sub_2FD7440((__int64)v83, v51, v93, a6);
                }
                if ( i != &v111 )
                  _libc_free((unsigned __int64)i);
                sub_C7D6A0(v106, 12LL * v108, 4);
              }
              ++v47;
            }
            while ( v97 != v47 );
          }
          v13 = 1;
          goto LABEL_83;
        }
        if ( (_BYTE)v13 )
          goto LABEL_132;
        LOBYTE(v13) = v66 != 0;
      }
LABEL_83:
      if ( v120 != v122 )
        _libc_free((unsigned __int64)v120);
      goto LABEL_85;
    }
  }
  if ( *((_BYTE *)v7 + 56) && (_BYTE)v13 )
    goto LABEL_77;
LABEL_85:
  if ( v117 != (__int64 *)v119 )
    _libc_free((unsigned __int64)v117);
  sub_C7D6A0(v114, 8LL * (unsigned int)v116, 8);
LABEL_8:
  sub_C7D6A0(v102, 4LL * (unsigned int)v104, 4);
  return v13;
}
