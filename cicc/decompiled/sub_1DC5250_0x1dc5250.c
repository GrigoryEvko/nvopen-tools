// Function: sub_1DC5250
// Address: 0x1dc5250
//
__int64 __fastcall sub_1DC5250(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  int v8; // r10d
  __int64 *v9; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // r9
  __int64 *v14; // rbx
  _QWORD *v15; // r15
  int v16; // r12d
  __int64 *v17; // rax
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 *v20; // rax
  int v21; // edx
  __int64 *v22; // r8
  __int64 **v23; // rcx
  int v24; // r13d
  __int64 v25; // rax
  void *v26; // rdx
  char *v27; // r13
  _QWORD *v28; // rax
  char *v29; // r12
  _QWORD *v30; // rbx
  __int64 **v31; // rax
  __int64 v32; // r8
  __int64 *v33; // rax
  __int64 v34; // rcx
  _QWORD *v35; // r9
  char *v36; // r15
  unsigned int v38; // esi
  __int64 v39; // r8
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned int *v45; // r15
  __int64 v46; // r13
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rsi
  unsigned int v53; // ecx
  __int64 *v54; // rdx
  __int64 v55; // rdi
  __int64 v56; // rax
  __m128i *v57; // rax
  __int64 v58; // rax
  int v59; // edx
  int v60; // r11d
  __int64 *v61; // rbx
  int v62; // eax
  int v63; // eax
  __int64 *v64; // rcx
  __int64 v65; // r13
  int v66; // ecx
  int v67; // r8d
  int v68; // r9d
  int v69; // r9d
  __int64 v70; // rdi
  unsigned int v71; // r13d
  __int64 *v72; // rdx
  __int64 v73; // rsi
  int v74; // r10d
  int v75; // r10d
  unsigned int v76; // edx
  __int64 v77; // rdi
  int v78; // esi
  __int128 v79; // [rsp-20h] [rbp-2D0h]
  int v80; // [rsp+14h] [rbp-29Ch]
  unsigned int nmemb; // [rsp+28h] [rbp-288h]
  __int64 *nmemba; // [rsp+28h] [rbp-288h]
  unsigned __int8 v85; // [rsp+40h] [rbp-270h]
  _QWORD *v86; // [rsp+40h] [rbp-270h]
  __int64 *v88; // [rsp+50h] [rbp-260h]
  unsigned int *v89; // [rsp+50h] [rbp-260h]
  __int64 *v90; // [rsp+58h] [rbp-258h]
  __int64 v91; // [rsp+58h] [rbp-258h]
  void *base; // [rsp+80h] [rbp-230h] BYREF
  __int64 v93; // [rsp+88h] [rbp-228h]
  _DWORD v94[16]; // [rsp+90h] [rbp-220h] BYREF
  __m128i v95; // [rsp+D0h] [rbp-1E0h] BYREF
  __m128i v96; // [rsp+E0h] [rbp-1D0h] BYREF
  _BYTE *v97; // [rsp+F0h] [rbp-1C0h]
  __int64 v98; // [rsp+F8h] [rbp-1B8h]
  _BYTE v99[432]; // [rsp+100h] [rbp-1B0h] BYREF

  v8 = 0;
  v9 = 0;
  v11 = *(unsigned int *)(a3 + 48);
  v80 = v11;
  base = v94;
  v93 = 0x1000000001LL;
  v94[0] = v11;
  nmemb = 0;
  v85 = 1;
  while ( 2 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8 * v11);
    v13 = *(__int64 **)(v12 + 72);
    v14 = *(__int64 **)(v12 + 64);
    LOBYTE(v12) = v14 == v13;
    v8 |= v12;
    if ( v14 == v13 )
      goto LABEL_17;
    v90 = v13;
    v13 = v9;
    v15 = a1;
    v16 = v8;
    do
    {
      while ( 1 )
      {
        v18 = *v14;
        v19 = *(unsigned int *)(*v14 + 48);
        if ( (*(_QWORD *)(v15[5] + 8LL * (*(_DWORD *)(*v14 + 48) >> 6)) & (1LL << v19)) != 0 )
        {
          v17 = *(__int64 **)(v15[12] + 16 * v19);
          if ( v17 )
          {
            if ( v17 == v13 || !v13 )
            {
              v13 = *(__int64 **)(v15[12] + 16 * v19);
            }
            else
            {
              v85 = 0;
              v13 = *(__int64 **)(v15[12] + 16 * v19);
            }
          }
          goto LABEL_8;
        }
        v88 = v13;
        v20 = sub_1DB7C30(
                a2,
                a7,
                a8,
                *(_QWORD *)(*(_QWORD *)(v15[2] + 392LL) + 16 * v19),
                *(_QWORD *)(*(_QWORD *)(v15[2] + 392LL) + 16 * v19 + 8));
        v22 = &qword_4FC4510;
        v13 = v88;
        if ( !(_BYTE)v21 )
          v22 = v20;
        *(_QWORD *)(v15[5] + 8LL * (*(_DWORD *)(v18 + 48) >> 6)) |= 1LL << *(_DWORD *)(v18 + 48);
        v23 = (__int64 **)(v15[12] + 16LL * *(unsigned int *)(v18 + 48));
        *v23 = v22;
        v23[1] = 0;
        if ( !v20 )
        {
          if ( (_BYTE)v21 )
          {
            v16 |= v21;
          }
          else if ( a3 == v18 )
          {
            a4 = 0;
          }
          else
          {
            v24 = *(_DWORD *)(v18 + 48);
            v25 = (unsigned int)v93;
            if ( (unsigned int)v93 >= HIDWORD(v93) )
            {
              sub_16CD150((__int64)&base, v94, 0, 4, (int)v22, (int)v88);
              v25 = (unsigned int)v93;
              v13 = v88;
            }
            *((_DWORD *)base + v25) = v24;
            LODWORD(v93) = v93 + 1;
          }
          goto LABEL_8;
        }
        if ( v88 && v88 != v20 )
          break;
        v13 = v20;
        v16 |= v21;
LABEL_8:
        if ( v90 == ++v14 )
          goto LABEL_16;
      }
      v85 = 0;
      v13 = v20;
      v16 |= v21;
      ++v14;
    }
    while ( v90 != v14 );
LABEL_16:
    v8 = v16;
    a1 = v15;
    v9 = v13;
LABEL_17:
    if ( ++nmemb != (_DWORD)v93 )
    {
      v11 = *((unsigned int *)base + nmemb);
      continue;
    }
    break;
  }
  *((_DWORD *)a1 + 36) = 0;
  if ( !((unsigned __int8)v8 | (v9 == &qword_4FC4510 || v9 == 0)) || !a8 )
  {
    if ( nmemb <= 4uLL )
      goto LABEL_31;
    goto LABEL_69;
  }
  if ( nmemb <= 4uLL )
    goto LABEL_46;
  v85 = 0;
LABEL_69:
  qsort(base, nmemb, 4u, (__compar_fn_t)sub_1DC3280);
LABEL_31:
  if ( !v85 )
  {
LABEL_46:
    v38 = *((_DWORD *)a1 + 22);
    LODWORD(v39) = (_DWORD)a1 + 64;
    if ( v38 )
    {
      v40 = a1[9];
      LODWORD(v41) = (v38 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
      v42 = (__int64 *)(v40 + 56LL * (unsigned int)v41);
      v43 = *v42;
      if ( a2 == *v42 )
      {
LABEL_48:
        nmemba = v42 + 1;
        v86 = v42 + 4;
LABEL_49:
        v44 = (unsigned int)v93;
        if ( *((_DWORD *)a1 + 37) < (unsigned int)v93 )
        {
          sub_16CD150((__int64)(a1 + 17), a1 + 19, (unsigned int)v93, 32, v39, (int)v13);
          v44 = (unsigned int)v93;
        }
        v36 = (char *)base + 4 * v44;
        if ( base == v36 )
        {
          v85 = 0;
          goto LABEL_42;
        }
        v89 = (unsigned int *)((char *)base + 4 * v44);
        v45 = (unsigned int *)base;
        while ( 1 )
        {
          while ( 1 )
          {
            v46 = *(_QWORD *)(*(_QWORD *)(*a1 + 96LL) + 8LL * *v45);
            if ( !a8 || (unsigned __int8)sub_1DC4D00(a1, a2, a7, a8, v46, nmemba, v86) )
              break;
LABEL_53:
            if ( v89 == ++v45 )
              goto LABEL_64;
          }
          v91 = a1[3];
          sub_1E06620(v91);
          v49 = 0;
          v50 = *(_QWORD *)(v91 + 1312);
          v51 = *(unsigned int *)(v50 + 48);
          if ( (_DWORD)v51 )
          {
            v52 = *(_QWORD *)(v50 + 32);
            v48 = v51 - 1;
            v53 = (v51 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
            v54 = (__int64 *)(v52 + 16LL * v53);
            v55 = *v54;
            if ( v46 == *v54 )
            {
LABEL_58:
              if ( v54 != (__int64 *)(v52 + 16 * v51) )
              {
                v49 = v54[1];
                goto LABEL_60;
              }
            }
            else
            {
              v59 = 1;
              while ( v55 != -8 )
              {
                v47 = v59 + 1;
                v53 = v48 & (v59 + v53);
                v54 = (__int64 *)(v52 + 16LL * v53);
                v55 = *v54;
                if ( v46 == *v54 )
                  goto LABEL_58;
                v59 = v47;
              }
            }
            v49 = 0;
          }
LABEL_60:
          v95.m128i_i64[1] = v49;
          v96 = 0u;
          v95.m128i_i64[0] = a2;
          v56 = *((unsigned int *)a1 + 36);
          if ( (unsigned int)v56 >= *((_DWORD *)a1 + 37) )
          {
            sub_16CD150((__int64)(a1 + 17), a1 + 19, 0, 32, v47, v48);
            v56 = *((unsigned int *)a1 + 36);
          }
          v57 = (__m128i *)(a1[17] + 32 * v56);
          *v57 = _mm_loadu_si128(&v95);
          v57[1] = _mm_loadu_si128(&v96);
          v58 = (unsigned int)(*((_DWORD *)a1 + 36) + 1);
          *((_DWORD *)a1 + 36) = v58;
          if ( a3 != v46 )
            goto LABEL_53;
          ++v45;
          *(_QWORD *)(a1[17] + 32 * v58 - 16) = a4;
          if ( v89 == v45 )
          {
LABEL_64:
            v85 = 0;
            v36 = (char *)base;
            goto LABEL_42;
          }
        }
      }
      v60 = 1;
      v61 = 0;
      while ( v43 != -8 )
      {
        if ( !v61 && v43 == -16 )
          v61 = v42;
        LODWORD(v13) = v60 + 1;
        v41 = (v38 - 1) & ((_DWORD)v41 + v60);
        v42 = (__int64 *)(v40 + 56 * v41);
        v43 = *v42;
        if ( a2 == *v42 )
          goto LABEL_48;
        ++v60;
      }
      if ( !v61 )
        v61 = v42;
      v62 = *((_DWORD *)a1 + 20);
      ++a1[8];
      v63 = v62 + 1;
      if ( 4 * v63 < 3 * v38 )
      {
        LODWORD(v64) = v38 >> 3;
        if ( v38 - *((_DWORD *)a1 + 21) - v63 > v38 >> 3 )
        {
LABEL_79:
          *((_DWORD *)a1 + 20) = v63;
          if ( *v61 != -8 )
            --*((_DWORD *)a1 + 21);
          *((_DWORD *)v61 + 6) = 0;
          v61[1] = 0;
          *v61 = a2;
          v61[2] = 0;
          v61[4] = 0;
          v61[5] = 0;
          *((_DWORD *)v61 + 12) = 0;
          v65 = (__int64)(*(_QWORD *)(*a1 + 104LL) - *(_QWORD *)(*a1 + 96LL)) >> 3;
          nmemba = v61 + 1;
          sub_13A49F0((__int64)(v61 + 1), v65, 0, (int)v64, v39, (int)v13);
          v86 = v61 + 4;
          sub_13A49F0((__int64)(v61 + 4), v65, 0, v66, v67, v68);
          goto LABEL_49;
        }
        sub_1DC4860((__int64)(a1 + 8), v38);
        v69 = *((_DWORD *)a1 + 22);
        if ( v69 )
        {
          LODWORD(v13) = v69 - 1;
          v70 = a1[9];
          LODWORD(v64) = 1;
          v71 = (unsigned int)v13 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
          v72 = 0;
          v61 = (__int64 *)(v70 + 56LL * v71);
          v73 = *v61;
          v63 = *((_DWORD *)a1 + 20) + 1;
          if ( a2 != *v61 )
          {
            while ( v73 != -8 )
            {
              if ( v73 == -16 && !v72 )
                v72 = v61;
              LODWORD(v39) = (_DWORD)v64 + 1;
              v71 = (unsigned int)v13 & ((_DWORD)v64 + v71);
              v61 = (__int64 *)(v70 + 56LL * v71);
              v73 = *v61;
              if ( a2 == *v61 )
                goto LABEL_79;
              LODWORD(v64) = (_DWORD)v64 + 1;
            }
            if ( v72 )
              v61 = v72;
          }
          goto LABEL_79;
        }
LABEL_111:
        ++*((_DWORD *)a1 + 20);
        BUG();
      }
    }
    else
    {
      ++a1[8];
    }
    sub_1DC4860((__int64)(a1 + 8), 2 * v38);
    v74 = *((_DWORD *)a1 + 22);
    if ( v74 )
    {
      v75 = v74 - 1;
      v39 = a1[9];
      v76 = v75 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      LODWORD(v64) = v76;
      v61 = (__int64 *)(v39 + 56LL * v76);
      v77 = *v61;
      v63 = *((_DWORD *)a1 + 20) + 1;
      if ( a2 != *v61 )
      {
        v78 = 1;
        v64 = 0;
        while ( v77 != -8 )
        {
          if ( !v64 && v77 == -16 )
            v64 = v61;
          LODWORD(v13) = v78 + 1;
          v76 = v75 & (v78 + v76);
          v61 = (__int64 *)(v39 + 56LL * v76);
          v77 = *v61;
          if ( a2 == *v61 )
            goto LABEL_79;
          ++v78;
        }
        if ( v64 )
          v61 = v64;
      }
      goto LABEL_79;
    }
    goto LABEL_111;
  }
  v26 = base;
  v95 = (__m128i)(unsigned __int64)a2;
  v27 = (char *)base;
  v97 = v99;
  v98 = 0x1000000000LL;
  if ( base != (char *)base + 4 * (unsigned int)v93 )
  {
    v28 = a1;
    v29 = (char *)base + 4 * (unsigned int)v93;
    v30 = v28;
    do
    {
      v32 = *(unsigned int *)v27;
      v33 = (__int64 *)(*(_QWORD *)(v30[2] + 392LL) + 16 * v32);
      v34 = *v33;
      v35 = (_QWORD *)v33[1];
      if ( v80 == (_DWORD)v32 && (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v35 = (_QWORD *)a4;
      }
      else
      {
        v31 = (__int64 **)(v30[12] + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v30 + 96LL) + 8 * v32) + 48LL));
        *v31 = v9;
        v31[1] = 0;
      }
      v27 += 4;
      *((_QWORD *)&v79 + 1) = v35;
      *(_QWORD *)&v79 = v34;
      sub_1DB8AC0((__int64)&v95, nmemb, (__int64)v26, v34, v32, v35, v79, (__int64)v9);
    }
    while ( v29 != v27 );
  }
  sub_1DB55F0((__int64)&v95);
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  v36 = (char *)base;
LABEL_42:
  if ( v36 != (char *)v94 )
    _libc_free((unsigned __int64)v36);
  return v85;
}
