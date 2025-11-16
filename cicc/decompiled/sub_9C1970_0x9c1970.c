// Function: sub_9C1970
// Address: 0x9c1970
//
__int64 __fastcall sub_9C1970(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        void (__fastcall *a7)(__int64),
        __int64 a8,
        void (__fastcall *a9)(__int64, _QWORD, _QWORD, __int64, _QWORD, __int64),
        __int64 a10,
        void (__fastcall *a11)(__int64, __int64, __int64, __int64, _QWORD, __int64),
        __int64 a12)
{
  __int64 v12; // r15
  char *v13; // rax
  unsigned int v14; // r14d
  char *v15; // rcx
  __int64 v16; // rbx
  _BYTE *v17; // r13
  int v18; // r12d
  __int64 v19; // rax
  _BYTE *v21; // r14
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // r11
  unsigned int v29; // r8d
  int v30; // r14d
  __int64 v31; // rbx
  __int64 v32; // r10
  int v33; // ecx
  signed int v34; // r13d
  unsigned int v35; // r9d
  __int64 v36; // r13
  __int64 v37; // r15
  __int64 v38; // r10
  char **v39; // r11
  char *v40; // r8
  signed __int64 v41; // rdi
  char *v42; // rax
  int v43; // edx
  __int64 v44; // r9
  __int64 v45; // rcx
  char *v46; // rdx
  unsigned int v47; // eax
  __int64 result; // rax
  _QWORD *v49; // r12
  __int64 v50; // rbx
  _QWORD *v51; // r14
  _QWORD *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // r13
  __int64 v55; // r12
  __int64 v56; // rax
  __int64 v57; // r14
  int v58; // edx
  __int64 v59; // r15
  unsigned __int64 v60; // rbx
  __int64 v61; // rsi
  __int64 v62; // r14
  __int64 i; // rax
  int v64; // edx
  __int64 v65; // rax
  unsigned int v66; // r15d
  __int64 v67; // rcx
  __int64 v68; // r13
  __int64 v69; // rbx
  __int64 j; // rdx
  int v71; // esi
  __int64 k; // rax
  __int64 v73; // rbx
  unsigned __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // r13
  __int64 v77; // rdi
  unsigned __int64 v78; // r8
  unsigned __int64 v79; // r13
  _QWORD *v80; // rbx
  _QWORD *v81; // r13
  int v82; // eax
  _QWORD *v83; // rbx
  _QWORD *v84; // r13
  __int64 v85; // rax
  unsigned __int64 v86; // [rsp+0h] [rbp-1B0h]
  int v88; // [rsp+40h] [rbp-170h]
  int v89; // [rsp+44h] [rbp-16Ch]
  __int64 v90; // [rsp+48h] [rbp-168h]
  unsigned int v91; // [rsp+58h] [rbp-158h]
  unsigned __int64 v93; // [rsp+60h] [rbp-150h]
  __int64 v94; // [rsp+68h] [rbp-148h]
  __int64 v95; // [rsp+68h] [rbp-148h]
  _QWORD *v96; // [rsp+70h] [rbp-140h]
  _QWORD *v97; // [rsp+70h] [rbp-140h]
  __int64 v98; // [rsp+70h] [rbp-140h]
  unsigned int v99; // [rsp+78h] [rbp-138h]
  unsigned int v100; // [rsp+78h] [rbp-138h]
  int v101; // [rsp+78h] [rbp-138h]
  unsigned int v102; // [rsp+80h] [rbp-130h]
  unsigned int v103; // [rsp+80h] [rbp-130h]
  int v104; // [rsp+84h] [rbp-12Ch]
  __int64 n; // [rsp+90h] [rbp-120h]
  unsigned __int64 v106; // [rsp+98h] [rbp-118h]
  __int64 v107; // [rsp+98h] [rbp-118h]
  __int64 v108; // [rsp+A0h] [rbp-110h]
  __int64 v109; // [rsp+A0h] [rbp-110h]
  unsigned int v110; // [rsp+B8h] [rbp-F8h]
  char **v111; // [rsp+B8h] [rbp-F8h]
  __int64 v112; // [rsp+B8h] [rbp-F8h]
  __int64 v114; // [rsp+C0h] [rbp-F0h]
  char **v115; // [rsp+C0h] [rbp-F0h]
  char **v116; // [rsp+C0h] [rbp-F0h]
  __int64 v117; // [rsp+C0h] [rbp-F0h]
  int v118; // [rsp+C8h] [rbp-E8h]
  __int64 v119; // [rsp+C8h] [rbp-E8h]
  unsigned int v120; // [rsp+C8h] [rbp-E8h]
  __int64 v121; // [rsp+C8h] [rbp-E8h]
  unsigned int v122; // [rsp+C8h] [rbp-E8h]
  __int64 v123; // [rsp+C8h] [rbp-E8h]
  char **v124; // [rsp+C8h] [rbp-E8h]
  __int64 v125; // [rsp+D8h] [rbp-D8h] BYREF
  _BYTE *v126; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v127; // [rsp+E8h] [rbp-C8h]
  _BYTE v128[48]; // [rsp+F0h] [rbp-C0h] BYREF
  _BYTE *v129; // [rsp+120h] [rbp-90h] BYREF
  __int64 v130; // [rsp+128h] [rbp-88h]
  _BYTE v131[80]; // [rsp+130h] [rbp-80h] BYREF
  char v132; // [rsp+180h] [rbp-30h] BYREF

  v12 = a4;
  v130 = 0x100000000LL;
  v13 = v131;
  v129 = v131;
  if ( a4 )
  {
    v14 = a2;
    v15 = &v132;
    v16 = 80 * v12;
    if ( v12 == 1
      || (a2 = v12,
          sub_9C1710((__int64)&v129, v12),
          v17 = v129,
          v15 = &v129[v16],
          v13 = &v129[80 * (unsigned int)v130],
          v13 != &v129[v16]) )
    {
      do
      {
        if ( v13 )
        {
          *((_DWORD *)v13 + 2) = 0;
          *(_QWORD *)v13 = v13 + 16;
          *((_DWORD *)v13 + 3) = 1;
        }
        v13 += 80;
      }
      while ( v15 != v13 );
      v17 = v129;
    }
    v91 = v14;
    v104 = 0;
    LODWORD(v130) = a4;
    v108 = 0;
    v88 = v14;
    v18 = v14 / a4;
    v110 = v14 / a3;
    v89 = 2 * a3;
    v106 = 2 * a3;
    v86 = v106 << 6;
    v118 = 2 * v14;
    v93 = v14 / a4;
    n = 4 * v93;
    v19 = 5 * v12;
    v90 = 16 * v19;
    do
    {
      v21 = &v17[v108];
      v126 = v128;
      v127 = 0xC00000000LL;
      if ( v106 > *(unsigned int *)&v17[v108 + 12] )
      {
        a2 = (__int64)(v21 + 16);
        v75 = 2 * a3;
        v98 = sub_C8D7D0(v21, v21 + 16, v106, 64, &v125);
        v76 = v98;
        do
        {
          while ( 1 )
          {
            if ( v76 )
            {
              *(_DWORD *)(v76 + 8) = 0;
              *(_QWORD *)v76 = v76 + 16;
              *(_DWORD *)(v76 + 12) = 12;
              if ( (_DWORD)v127 )
                break;
            }
            v76 += 64;
            if ( !--v75 )
              goto LABEL_115;
          }
          a2 = (__int64)&v126;
          v77 = v76;
          v76 += 64;
          sub_9B6660(v77, (__int64)&v126);
          --v75;
        }
        while ( v75 );
LABEL_115:
        v78 = (unsigned __int64)*((unsigned int *)v21 + 2) << 6;
        v79 = *(_QWORD *)v21 + v78;
        if ( *(_QWORD *)v21 != v79 )
        {
          v80 = (_QWORD *)(*(_QWORD *)v21 + v78);
          v81 = *(_QWORD **)v21;
          do
          {
            v80 -= 8;
            if ( (_QWORD *)*v80 != v80 + 2 )
              _libc_free(*v80, a2);
          }
          while ( v81 != v80 );
          v79 = *(_QWORD *)v21;
        }
        v82 = v125;
        if ( v21 + 16 != (_BYTE *)v79 )
        {
          v101 = v125;
          _libc_free(v79, a2);
          v82 = v101;
        }
        *((_DWORD *)v21 + 3) = v82;
        *(_QWORD *)v21 = v98;
        *((_DWORD *)v21 + 2) = v89;
      }
      else
      {
        v22 = *((unsigned int *)v21 + 2);
        v23 = 2 * a3;
        if ( v106 > v22 )
          v23 = *((unsigned int *)v21 + 2);
        if ( v23 )
        {
          v24 = *(_QWORD *)v21;
          v25 = *(_QWORD *)v21 + (v23 << 6);
          do
          {
            v26 = v24;
            a2 = (__int64)&v126;
            v24 += 64;
            sub_9B6660(v26, (__int64)&v126);
          }
          while ( v25 != v24 );
          v21 = &v17[v108];
          v22 = *(unsigned int *)&v17[v108 + 8];
        }
        if ( v106 > v22 )
        {
          v73 = *(_QWORD *)v21 + (v22 << 6);
          v74 = v106 - v22;
          if ( v106 != v22 )
          {
            do
            {
              if ( v73 )
              {
                *(_DWORD *)(v73 + 8) = 0;
                *(_QWORD *)v73 = v73 + 16;
                *(_DWORD *)(v73 + 12) = 12;
                if ( (_DWORD)v127 )
                {
                  a2 = (__int64)&v126;
                  sub_9B6660(v73, (__int64)&v126);
                }
              }
              v73 += 64;
              --v74;
            }
            while ( v74 );
          }
        }
        else if ( v106 < v22 && *(_QWORD *)v21 + (v22 << 6) != *(_QWORD *)v21 + v86 )
        {
          v83 = (_QWORD *)(*(_QWORD *)v21 + (v22 << 6));
          v84 = (_QWORD *)(*(_QWORD *)v21 + v86);
          do
          {
            v83 -= 8;
            if ( (_QWORD *)*v83 != v83 + 2 )
              _libc_free(*v83, a2);
          }
          while ( v84 != v83 );
        }
        *((_DWORD *)v21 + 2) = v89;
      }
      if ( v126 != v128 )
        _libc_free(v126, a2);
      if ( a4 <= v91 )
      {
        v27 = v104;
        if ( v104 != v88 )
        {
          v28 = v21;
          v29 = v18;
          v30 = v104;
          v31 = 0;
          v32 = a1;
          do
          {
            v33 = *(_DWORD *)(v32 + 4 * v27);
            if ( v33 != -1 && v33 < v118 )
            {
              v34 = v33 % v88 / v110 + a3;
              v35 = v33 % v88 % v110;
              if ( v33 < v88 )
                v34 = v33 % v88 / v110;
              v36 = (__int64)v34 << 6;
              v37 = v36 + *v28;
              if ( !*(_DWORD *)(v37 + 8) )
              {
                if ( v93 > *(unsigned int *)(v37 + 12) )
                {
                  v100 = v33 % v88 % v110;
                  v103 = v29;
                  v95 = v32;
                  v97 = v28;
                  sub_C8D5F0(v37, v37 + 16, v93, 4);
                  a2 = 255;
                  memset(*(void **)v37, 255, n);
                  v29 = v103;
                  v35 = v100;
                  v28 = v97;
                  v32 = v95;
                  *(_DWORD *)(v37 + 8) = v103;
                }
                else
                {
                  if ( n )
                  {
                    v99 = v33 % v88 % v110;
                    a2 = 255;
                    v102 = v29;
                    v94 = v32;
                    v96 = v28;
                    memset(*(void **)v37, 255, n);
                    v35 = v99;
                    v28 = v96;
                    v32 = v94;
                    v29 = v102;
                  }
                  *(_DWORD *)(v37 + 8) = v29;
                }
                v37 = v36 + *v28;
              }
              *(_DWORD *)(*(_QWORD *)v37 + 4 * v31) = v35;
            }
            if ( v29 <= (int)v31 + 1 )
              break;
            ++v30;
            ++v31;
            v27 = v30;
          }
          while ( v88 != v30 );
          a1 = v32;
          v18 = v29;
        }
      }
      v108 += 80;
      v17 = v129;
      v104 += v18;
    }
    while ( v90 != v108 );
    v107 = a5;
    if ( !a5 )
      goto LABEL_55;
  }
  else
  {
    result = a5;
    v17 = v131;
    v107 = a5;
    if ( !a5 )
      return result;
  }
  v109 = 0;
  v38 = 2 * a3;
  do
  {
    v39 = (char **)&v17[80 * (unsigned int)v109];
    v40 = *v39;
    v41 = (unsigned __int64)*((unsigned int *)v39 + 2) << 6;
    a2 = (__int64)&(*v39)[v41];
    if ( *v39 != (char *)a2 )
    {
      v42 = *v39;
      v43 = 0;
      do
      {
        v43 -= (*((_DWORD *)v42 + 2) == 0) - 1;
        v42 += 64;
      }
      while ( (char *)a2 != v42 );
      if ( v43 )
      {
        v44 = 1;
        if ( v43 == 1 )
        {
          v45 = v41 >> 6;
          if ( (unsigned __int8)((__int64)*((unsigned int *)v39 + 2) >> 2) )
          {
            v46 = *v39;
            while ( 1 )
            {
              v47 = *((_DWORD *)v46 + 2);
              if ( v47 )
                goto LABEL_52;
              v47 = *((_DWORD *)v46 + 18);
              if ( v47 )
              {
                v46 += 64;
                v45 = (v46 - v40) >> 6;
                goto LABEL_53;
              }
              v47 = *((_DWORD *)v46 + 34);
              if ( v47 )
              {
                v46 += 128;
                v45 = (v46 - v40) >> 6;
                goto LABEL_53;
              }
              v47 = *((_DWORD *)v46 + 50);
              if ( v47 )
              {
                v46 += 192;
                v45 = (v46 - v40) >> 6;
                goto LABEL_53;
              }
              v46 += 256;
              if ( &v40[256 * (__int64)(char)((__int64)*((unsigned int *)v39 + 2) >> 2)] == v46 )
              {
                v85 = (a2 - (__int64)v46) >> 6;
                goto LABEL_132;
              }
            }
          }
          v85 = v41 >> 6;
          v46 = *v39;
LABEL_132:
          if ( v85 != 2 )
          {
            if ( v85 != 3 )
            {
              if ( v85 != 1 )
              {
                v47 = *(_DWORD *)(a2 + 8);
                v46 = &(*v39)[v41];
                goto LABEL_53;
              }
              goto LABEL_142;
            }
            v47 = *((_DWORD *)v46 + 2);
            if ( v47 )
            {
LABEL_52:
              v45 = (v46 - v40) >> 6;
LABEL_53:
              v119 = v38;
              a2 = *(_QWORD *)v46;
              a9(a10, *(_QWORD *)v46, v47, v45, (unsigned int)v109, 1);
              v38 = v119;
              goto LABEL_54;
            }
            v46 += 64;
          }
          v47 = *((_DWORD *)v46 + 2);
          if ( !v47 )
          {
            v46 += 64;
LABEL_142:
            v47 = *((_DWORD *)v46 + 2);
            if ( !v47 )
            {
              v47 = *(_DWORD *)(a2 + 8);
              v46 = &(*v39)[v41];
              goto LABEL_53;
            }
            goto LABEL_52;
          }
          goto LABEL_52;
        }
LABEL_69:
        if ( !v38 )
          goto LABEL_54;
        v53 = 0xFFFFFFFFLL;
        v54 = 0;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        v58 = -1;
        v59 = 0;
        while ( 1 )
        {
          v60 = (unsigned __int64)&(*v39)[64 * (unsigned __int64)(unsigned int)v54];
          a2 = *(unsigned int *)(v60 + 8);
          if ( !(_DWORD)a2 )
            goto LABEL_72;
          if ( (_DWORD)v53 == v58 )
          {
            v55 = *(_QWORD *)v60;
            v53 = (unsigned int)v54;
            v59 = (unsigned int)a2;
LABEL_72:
            if ( v38 == ++v54 )
              goto LABEL_87;
          }
          else
          {
            if ( (int)v59 <= 0 )
            {
              v112 = v38;
              a2 = v55;
              v116 = v39;
              v122 = v53;
              a11(a12, v55, v59, v53, (unsigned int)v54, v44);
              v53 = v122;
              v39 = v116;
              v38 = v112;
            }
            else
            {
              v61 = *(_QWORD *)v60;
              v62 = (unsigned int)(v59 - 1);
              for ( i = 0; ; ++i )
              {
                v64 = *(_DWORD *)(v61 + 4 * i);
                if ( v64 != -1 )
                  *(_DWORD *)(v55 + 4 * i) = v59 + v64;
                if ( v62 == i )
                  break;
              }
              v111 = v39;
              a2 = v55;
              v114 = v38;
              v120 = v53;
              a11(a12, v55, v59, v53, (unsigned int)v54, v44);
              v53 = v120;
              v39 = v111;
              v65 = 0;
              v38 = v114;
              while ( 1 )
              {
                if ( *(_DWORD *)(v55 + 4 * v65) != -1 )
                  *(_DWORD *)(v55 + 4 * v65) = v65;
                if ( v62 == v65 )
                  break;
                ++v65;
              }
            }
            ++v54;
            *(_DWORD *)(v60 + 8) = 0;
            v56 = v59;
            v58 = v53;
            v57 = v55;
            v44 = 0;
            if ( v38 == v54 )
            {
LABEL_87:
              v66 = v53;
              v67 = (unsigned int)v58;
              if ( v58 == v66 )
              {
                if ( v58 < 0 )
                  goto LABEL_54;
              }
              else
              {
                if ( v58 < 0 )
                  goto LABEL_54;
                v68 = (__int64)(int)v66 << 6;
                if ( (int)v56 <= 0 )
                {
                  v117 = v38;
                  a2 = v57;
                  v124 = v39;
                  a11(a12, v57, v56, (unsigned int)v58, v66, v44);
                  v39 = v124;
                  v38 = v117;
                  v44 = 0;
                  *(_DWORD *)&(*v124)[v68 + 8] = 0;
                }
                else
                {
                  v69 = (unsigned int)(v56 - 1);
                  for ( j = 0; ; ++j )
                  {
                    v71 = *(_DWORD *)(v55 + 4 * j);
                    if ( v71 != -1 )
                      *(_DWORD *)(v57 + 4 * j) = v56 + v71;
                    if ( v69 == j )
                      break;
                  }
                  v115 = v39;
                  a2 = v57;
                  v121 = v38;
                  a11(a12, v57, v56, v67, v66, v44);
                  v39 = v115;
                  v38 = v121;
                  *(_DWORD *)&(*v115)[v68 + 8] = 0;
                  for ( k = 0; ; ++k )
                  {
                    if ( *(_DWORD *)(v57 + 4 * k) != -1 )
                      *(_DWORD *)(v57 + 4 * k) = k;
                    if ( k == v69 )
                      break;
                  }
                  v44 = 0;
                }
              }
              goto LABEL_69;
            }
          }
        }
      }
    }
    v123 = v38;
    a7(a8);
    v38 = v123;
LABEL_54:
    ++v109;
    v17 = v129;
  }
  while ( v109 != v107 );
LABEL_55:
  result = (unsigned int)v130;
  v49 = &v17[80 * (unsigned int)v130];
  if ( v49 != (_QWORD *)v17 )
  {
    do
    {
      v50 = *((unsigned int *)v49 - 18);
      v51 = (_QWORD *)*(v49 - 10);
      v49 -= 10;
      v52 = &v51[8 * v50];
      if ( v51 != v52 )
      {
        do
        {
          v52 -= 8;
          if ( (_QWORD *)*v52 != v52 + 2 )
            _libc_free(*v52, a2);
        }
        while ( v51 != v52 );
        v51 = (_QWORD *)*v49;
      }
      result = (__int64)(v49 + 2);
      if ( v51 != v49 + 2 )
        result = _libc_free(v51, a2);
    }
    while ( v49 != (_QWORD *)v17 );
    v17 = v129;
  }
  if ( v17 != v131 )
    return _libc_free(v17, a2);
  return result;
}
