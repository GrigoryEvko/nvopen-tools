// Function: sub_1459590
// Address: 0x1459590
//
__int64 __fastcall sub_1459590(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v4; // ecx
  __int64 v6; // rdi
  unsigned int v7; // eax
  int v8; // esi
  __int64 *v9; // rbx
  __int64 v10; // rdx
  unsigned __int64 v11; // rdi
  int v12; // eax
  int v13; // edx
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 *v16; // rbx
  __int64 v17; // rcx
  unsigned __int64 v18; // rdi
  int v19; // eax
  int v20; // edx
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 *v23; // rbx
  __int64 v24; // rcx
  unsigned __int64 v25; // rdi
  int v26; // eax
  int v27; // edx
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 *v30; // rbx
  __int64 v31; // rcx
  int v32; // eax
  int v33; // edx
  __int64 v34; // rsi
  unsigned int v35; // eax
  __int64 *v36; // rbx
  __int64 v37; // rcx
  int v38; // eax
  int v39; // edx
  __int64 v40; // rcx
  unsigned int v41; // eax
  __int64 *v42; // rbx
  __int64 v43; // rsi
  __int64 v44; // rdi
  int v45; // eax
  int v46; // ecx
  __int64 v47; // rsi
  unsigned int v48; // edx
  __int64 *v49; // rax
  __int64 v50; // rdi
  int v51; // eax
  int v52; // ecx
  __int64 v53; // rsi
  unsigned int v54; // edx
  __int64 *v55; // rax
  __int64 v56; // rdi
  __int64 result; // rax
  __int64 v58; // rsi
  _QWORD *v59; // rcx
  _QWORD *v60; // r14
  _QWORD *v61; // rdx
  __int64 v62; // rax
  _QWORD *v63; // r12
  _QWORD *v64; // rbx
  unsigned __int64 v65; // rdi
  __int64 *v66; // rbx
  __int64 *v67; // r14
  unsigned __int64 v68; // r12
  __int64 v69; // r15
  __int64 v70; // rcx
  _QWORD *v71; // rax
  _QWORD *v72; // rsi
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rdi
  __int64 *v75; // rbx
  __int64 *v76; // r15
  unsigned __int64 v77; // r12
  __int64 v78; // r14
  __int64 v79; // rcx
  _QWORD *v80; // rax
  _QWORD *v81; // rsi
  unsigned __int64 v82; // rdi
  unsigned __int64 v83; // rdi
  int v84; // edi
  int v85; // edi
  int v86; // edi
  int v87; // edi
  int v88; // edi
  int v89; // eax
  int v90; // r8d
  int v91; // eax
  int v92; // r8d
  _QWORD *v94; // [rsp+10h] [rbp-50h]
  _QWORD *v95; // [rsp+10h] [rbp-50h]
  __int64 v96; // [rsp+18h] [rbp-48h]
  __int64 v97; // [rsp+18h] [rbp-48h]
  _QWORD *v98; // [rsp+28h] [rbp-38h]
  _QWORD *v99; // [rsp+28h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 648);
  if ( v3 )
  {
    v4 = v3 - 1;
    v6 = *(_QWORD *)(a1 + 632);
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = 1;
    v9 = (__int64 *)(v6 + 56LL * v7);
    v10 = *v9;
    if ( *v9 == a2 )
    {
LABEL_3:
      v11 = v9[1];
      if ( (__int64 *)v11 != v9 + 3 )
        _libc_free(v11);
      *v9 = -16;
      --*(_DWORD *)(a1 + 640);
      ++*(_DWORD *)(a1 + 644);
    }
    else
    {
      while ( v10 != -8 )
      {
        v7 = v4 & (v8 + v7);
        v9 = (__int64 *)(v6 + 56LL * v7);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_3;
        ++v8;
      }
    }
  }
  v12 = *(_DWORD *)(a1 + 680);
  if ( v12 )
  {
    v13 = v12 - 1;
    v14 = *(_QWORD *)(a1 + 664);
    v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (__int64 *)(v14 + 40LL * v15);
    v17 = *v16;
    if ( *v16 == a2 )
    {
LABEL_8:
      v18 = v16[1];
      if ( (__int64 *)v18 != v16 + 3 )
        _libc_free(v18);
      *v16 = -16;
      --*(_DWORD *)(a1 + 672);
      ++*(_DWORD *)(a1 + 676);
    }
    else
    {
      v87 = 1;
      while ( v17 != -8 )
      {
        v15 = v13 & (v87 + v15);
        v16 = (__int64 *)(v14 + 40LL * v15);
        v17 = *v16;
        if ( *v16 == a2 )
          goto LABEL_8;
        ++v87;
      }
    }
  }
  v19 = *(_DWORD *)(a1 + 744);
  if ( v19 )
  {
    v20 = v19 - 1;
    v21 = *(_QWORD *)(a1 + 728);
    v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (__int64 *)(v21 + 40LL * v22);
    v24 = *v23;
    if ( *v23 == a2 )
    {
LABEL_13:
      v25 = v23[1];
      if ( (__int64 *)v25 != v23 + 3 )
        _libc_free(v25);
      *v23 = -16;
      --*(_DWORD *)(a1 + 736);
      ++*(_DWORD *)(a1 + 740);
    }
    else
    {
      v88 = 1;
      while ( v24 != -8 )
      {
        v22 = v20 & (v88 + v22);
        v23 = (__int64 *)(v21 + 40LL * v22);
        v24 = *v23;
        if ( *v23 == a2 )
          goto LABEL_13;
        ++v88;
      }
    }
  }
  v26 = *(_DWORD *)(a1 + 776);
  if ( v26 )
  {
    v27 = v26 - 1;
    v28 = *(_QWORD *)(a1 + 760);
    v29 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v30 = (__int64 *)(v28 + 40LL * v29);
    v31 = *v30;
    if ( *v30 == a2 )
    {
LABEL_18:
      sub_135E100(v30 + 3);
      sub_135E100(v30 + 1);
      *v30 = -16;
      --*(_DWORD *)(a1 + 768);
      ++*(_DWORD *)(a1 + 772);
    }
    else
    {
      v84 = 1;
      while ( v31 != -8 )
      {
        v29 = v27 & (v84 + v29);
        v30 = (__int64 *)(v28 + 40LL * v29);
        v31 = *v30;
        if ( *v30 == a2 )
          goto LABEL_18;
        ++v84;
      }
    }
  }
  v32 = *(_DWORD *)(a1 + 808);
  if ( v32 )
  {
    v33 = v32 - 1;
    v34 = *(_QWORD *)(a1 + 792);
    v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v36 = (__int64 *)(v34 + 40LL * v35);
    v37 = *v36;
    if ( *v36 == a2 )
    {
LABEL_21:
      sub_135E100(v36 + 3);
      sub_135E100(v36 + 1);
      *v36 = -16;
      --*(_DWORD *)(a1 + 800);
      ++*(_DWORD *)(a1 + 804);
    }
    else
    {
      v85 = 1;
      while ( v37 != -8 )
      {
        v35 = v33 & (v85 + v35);
        v36 = (__int64 *)(v34 + 40LL * v35);
        v37 = *v36;
        if ( *v36 == a2 )
          goto LABEL_21;
        ++v85;
      }
    }
  }
  v38 = *(_DWORD *)(a1 + 136);
  if ( v38 )
  {
    v39 = v38 - 1;
    v40 = *(_QWORD *)(a1 + 120);
    v41 = (v38 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v42 = (__int64 *)(v40 + ((unsigned __int64)v41 << 6));
    v43 = *v42;
    if ( *v42 == a2 )
    {
LABEL_24:
      v44 = v42[5];
      if ( v44 )
        j_j___libc_free_0(v44, v42[7] - v44);
      j___libc_free_0(v42[2]);
      *v42 = -16;
      --*(_DWORD *)(a1 + 128);
      ++*(_DWORD *)(a1 + 132);
    }
    else
    {
      v86 = 1;
      while ( v43 != -8 )
      {
        v41 = v39 & (v86 + v41);
        v42 = (__int64 *)(v40 + ((unsigned __int64)v41 << 6));
        v43 = *v42;
        if ( *v42 == a2 )
          goto LABEL_24;
        ++v86;
      }
    }
  }
  v45 = *(_DWORD *)(a1 + 104);
  if ( v45 )
  {
    v46 = v45 - 1;
    v47 = *(_QWORD *)(a1 + 88);
    v48 = (v45 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v49 = (__int64 *)(v47 + 16LL * v48);
    v50 = *v49;
    if ( *v49 == a2 )
    {
LABEL_29:
      *v49 = -16;
      --*(_DWORD *)(a1 + 96);
      ++*(_DWORD *)(a1 + 100);
    }
    else
    {
      v91 = 1;
      while ( v50 != -8 )
      {
        v92 = v91 + 1;
        v48 = v46 & (v91 + v48);
        v49 = (__int64 *)(v47 + 16LL * v48);
        v50 = *v49;
        if ( *v49 == a2 )
          goto LABEL_29;
        v91 = v92;
      }
    }
  }
  v51 = *(_DWORD *)(a1 + 520);
  if ( v51 )
  {
    v52 = v51 - 1;
    v53 = *(_QWORD *)(a1 + 504);
    v54 = (v51 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v55 = (__int64 *)(v53 + 16LL * v54);
    v56 = *v55;
    if ( *v55 == a2 )
    {
LABEL_32:
      *v55 = -16;
      --*(_DWORD *)(a1 + 512);
      ++*(_DWORD *)(a1 + 516);
    }
    else
    {
      v89 = 1;
      while ( v56 != -8 )
      {
        v90 = v89 + 1;
        v54 = v52 & (v89 + v54);
        v55 = (__int64 *)(v53 + 16LL * v54);
        v56 = *v55;
        if ( *v55 == a2 )
          goto LABEL_32;
        v89 = v90;
      }
    }
  }
  if ( !*(_DWORD *)(a1 + 1016) )
    goto LABEL_34;
  v58 = *(unsigned int *)(a1 + 1024);
  v59 = *(_QWORD **)(a1 + 1008);
  v60 = &v59[8 * v58];
  if ( v59 == v60 )
    goto LABEL_34;
  v61 = *(_QWORD **)(a1 + 1008);
  while ( 1 )
  {
    v62 = *v61;
    v63 = v61;
    if ( *v61 != -8 )
      break;
    if ( v61[1] != -8 )
      goto LABEL_41;
LABEL_139:
    v61 += 8;
    if ( v60 == v61 )
      goto LABEL_34;
  }
  if ( v62 == -16 && v61[1] == -16 )
    goto LABEL_139;
LABEL_41:
  if ( v60 != v61 )
  {
    while ( 1 )
    {
      v64 = v63 + 8;
      if ( v62 )
        v62 += 32;
      if ( v62 == a2 )
      {
        while ( v60 != v64 )
        {
          if ( *v64 == -8 )
          {
            if ( v64[1] != -8 )
              break;
          }
          else if ( *v64 != -16 || v64[1] != -16 )
          {
            break;
          }
          v64 += 8;
        }
        v65 = v63[3];
        if ( (_QWORD *)v65 != v63 + 5 )
          _libc_free(v65);
        *v63 = -16;
        v63[1] = -16;
        v59 = *(_QWORD **)(a1 + 1008);
        --*(_DWORD *)(a1 + 1016);
        v58 = *(unsigned int *)(a1 + 1024);
        ++*(_DWORD *)(a1 + 1020);
      }
      else
      {
        for ( ; v60 != v64; v64 += 8 )
        {
          if ( *v64 == -8 )
          {
            if ( v64[1] != -8 )
              break;
          }
          else if ( *v64 != -16 || v64[1] != -16 )
          {
            break;
          }
        }
      }
      if ( v64 == &v59[8 * v58] )
        break;
      v62 = *v64;
      v63 = v64;
    }
  }
LABEL_34:
  result = *(unsigned int *)(a1 + 544);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 536);
    v75 = (__int64 *)(result + ((unsigned __int64)*(unsigned int *)(a1 + 552) << 6));
    if ( (__int64 *)result != v75 )
    {
      while ( 1 )
      {
        v95 = (_QWORD *)result;
        if ( *(_QWORD *)result != -8 && *(_QWORD *)result != -16 )
          break;
        result += 64;
        if ( v75 == (__int64 *)result )
          goto LABEL_35;
      }
      if ( (__int64 *)result != v75 )
      {
        do
        {
          v76 = v95 + 8;
          result = sub_14594D0(v95 + 1, a2, a1);
          if ( (_BYTE)result )
          {
            sub_14575E0(v95 + 1);
            for ( ; v75 != v76; v76 += 8 )
            {
              if ( *v76 != -8 && *v76 != -16 )
                break;
            }
            v97 = v95[1];
            v77 = v97 + 24LL * *((unsigned int *)v95 + 4);
            if ( v97 != v77 )
            {
              do
              {
                v78 = *(_QWORD *)(v77 - 8);
                v77 -= 24LL;
                if ( v78 )
                {
                  v79 = *(unsigned int *)(v78 + 208);
                  *(_QWORD *)v78 = &unk_49EC708;
                  if ( (_DWORD)v79 )
                  {
                    v80 = *(_QWORD **)(v78 + 192);
                    v81 = &v80[7 * v79];
                    do
                    {
                      if ( *v80 != -8 && *v80 != -16 )
                      {
                        v82 = v80[1];
                        if ( (_QWORD *)v82 != v80 + 3 )
                        {
                          v99 = v80;
                          _libc_free(v82);
                          v80 = v99;
                        }
                      }
                      v80 += 7;
                    }
                    while ( v81 != v80 );
                  }
                  j___libc_free_0(*(_QWORD *)(v78 + 192));
                  v83 = *(_QWORD *)(v78 + 40);
                  if ( v83 != v78 + 56 )
                    _libc_free(v83);
                  j_j___libc_free_0(v78, 216);
                }
              }
              while ( v97 != v77 );
              v77 = v95[1];
            }
            if ( (_QWORD *)v77 != v95 + 3 )
              _libc_free(v77);
            result = (__int64)v95;
            v95 = v76;
            *(_QWORD *)result = -16;
            --*(_DWORD *)(a1 + 544);
            ++*(_DWORD *)(a1 + 548);
          }
          else
          {
            if ( v75 == v76 )
              break;
            while ( 1 )
            {
              result = *v76;
              if ( *v76 != -8 && result != -16 )
                break;
              v76 += 8;
              if ( v75 == v76 )
                goto LABEL_35;
            }
            v95 = v76;
          }
        }
        while ( v75 != v76 );
      }
    }
  }
LABEL_35:
  if ( *(_DWORD *)(a1 + 576) )
  {
    result = *(_QWORD *)(a1 + 568);
    v66 = (__int64 *)(result + ((unsigned __int64)*(unsigned int *)(a1 + 584) << 6));
    if ( (__int64 *)result != v66 )
    {
      while ( 1 )
      {
        v94 = (_QWORD *)result;
        if ( *(_QWORD *)result != -16 && *(_QWORD *)result != -8 )
          break;
        result += 64;
        if ( v66 == (__int64 *)result )
          return result;
      }
      if ( (__int64 *)result != v66 )
      {
        do
        {
          v67 = v94 + 8;
          result = sub_14594D0(v94 + 1, a2, a1);
          if ( (_BYTE)result )
          {
            sub_14575E0(v94 + 1);
            for ( ; v66 != v67; v67 += 8 )
            {
              if ( *v67 != -16 && *v67 != -8 )
                break;
            }
            v96 = v94[1];
            v68 = v96 + 24LL * *((unsigned int *)v94 + 4);
            if ( v96 != v68 )
            {
              do
              {
                v69 = *(_QWORD *)(v68 - 8);
                v68 -= 24LL;
                if ( v69 )
                {
                  v70 = *(unsigned int *)(v69 + 208);
                  *(_QWORD *)v69 = &unk_49EC708;
                  if ( (_DWORD)v70 )
                  {
                    v71 = *(_QWORD **)(v69 + 192);
                    v72 = &v71[7 * v70];
                    do
                    {
                      if ( *v71 != -8 && *v71 != -16 )
                      {
                        v73 = v71[1];
                        if ( (_QWORD *)v73 != v71 + 3 )
                        {
                          v98 = v71;
                          _libc_free(v73);
                          v71 = v98;
                        }
                      }
                      v71 += 7;
                    }
                    while ( v72 != v71 );
                  }
                  j___libc_free_0(*(_QWORD *)(v69 + 192));
                  v74 = *(_QWORD *)(v69 + 40);
                  if ( v74 != v69 + 56 )
                    _libc_free(v74);
                  j_j___libc_free_0(v69, 216);
                }
              }
              while ( v96 != v68 );
              v68 = v94[1];
            }
            if ( (_QWORD *)v68 != v94 + 3 )
              _libc_free(v68);
            result = (__int64)v94;
            v94 = v67;
            *(_QWORD *)result = -16;
            --*(_DWORD *)(a1 + 576);
            ++*(_DWORD *)(a1 + 580);
          }
          else
          {
            if ( v67 == v66 )
              return result;
            while ( 1 )
            {
              result = *v67;
              if ( *v67 != -16 && result != -8 )
                break;
              v67 += 8;
              if ( v66 == v67 )
                return result;
            }
            v94 = v67;
          }
        }
        while ( v66 != v67 );
      }
    }
  }
  return result;
}
