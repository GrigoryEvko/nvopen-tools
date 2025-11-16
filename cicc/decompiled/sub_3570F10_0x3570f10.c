// Function: sub_3570F10
// Address: 0x3570f10
//
__int64 __fastcall sub_3570F10(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 *v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // ebx
  __int64 v10; // r12
  __int64 i; // rbx
  int v12; // ecx
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  unsigned int v21; // esi
  __int64 v22; // r9
  int v23; // r11d
  _QWORD *v24; // rax
  unsigned int v25; // ecx
  _QWORD *v26; // rdx
  __int64 v27; // r8
  _DWORD *v28; // rax
  int v29; // eax
  unsigned __int64 v30; // r10
  __int64 v31; // r15
  _BYTE *v32; // rbx
  char v33; // r11
  _BYTE *v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // r9
  int v37; // eax
  __int64 v38; // rsi
  __int64 *v39; // rdx
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // edi
  int v44; // edx
  __int64 v45; // rdx
  int v46; // r14d
  int v47; // r14d
  __int64 v48; // r11
  __int64 v49; // rcx
  __int64 v50; // rdi
  int v51; // r10d
  _QWORD *v52; // rsi
  int v53; // r14d
  int v54; // r14d
  __int64 v55; // r11
  int v56; // r10d
  __int64 v57; // rcx
  __int64 v58; // rdi
  unsigned int v59; // esi
  __int64 v60; // rdx
  __int64 v61; // r11
  __int64 *v62; // rdi
  int v63; // r8d
  unsigned int v64; // ecx
  __int64 *v65; // rax
  __int64 v66; // r9
  __int64 v67; // rax
  int v68; // r9d
  int v69; // r9d
  __int64 v70; // r10
  int v71; // ecx
  unsigned int v72; // esi
  __int64 v73; // r8
  int v74; // edi
  __int64 *v75; // r11
  int v76; // edi
  int v77; // r9d
  int v78; // r9d
  __int64 v79; // r10
  int v80; // edi
  unsigned int v81; // esi
  __int64 v82; // r8
  unsigned __int64 v83; // [rsp+10h] [rbp-3A0h]
  __int64 v84; // [rsp+18h] [rbp-398h]
  _BYTE *v85; // [rsp+20h] [rbp-390h]
  __int64 v86; // [rsp+28h] [rbp-388h]
  char v87; // [rsp+33h] [rbp-37Dh]
  unsigned int v88; // [rsp+34h] [rbp-37Ch]
  __int64 v89; // [rsp+38h] [rbp-378h]
  __int64 v91; // [rsp+48h] [rbp-368h] BYREF
  _BYTE *v92; // [rsp+50h] [rbp-360h] BYREF
  __int64 v93; // [rsp+58h] [rbp-358h]
  _BYTE v94[848]; // [rsp+60h] [rbp-350h] BYREF

  v92 = v94;
  v91 = a2;
  v93 = 0x6400000000LL;
  v84 = sub_356EC70((__int64)a1, a2, (__int64)&v92);
  v2 = (unsigned int)v93;
  if ( !(_DWORD)v93 )
  {
    v3 = *a1;
    v4 = *(_QWORD *)(*a1 + 32);
    v5 = *(_QWORD *)(*a1 + 40);
    v6 = (__int64 *)sub_2E311E0(v91);
    sub_356E080(0xAu, v91, v6, *(_QWORD *)(v3 + 8), *(_QWORD *)(v3 + 16), v5, v4);
    v8 = *(_DWORD *)(*(_QWORD *)(v7 + 32) + 8LL);
    *(_DWORD *)sub_356E5F0(a1[1], &v91) = v8;
    goto LABEL_3;
  }
  do
  {
    v83 = (unsigned __int64)v92;
    v85 = &v92[8 * v2];
    if ( v92 == v85 )
    {
      v31 = (__int64)a1;
      goto LABEL_79;
    }
    v87 = 0;
    do
    {
      if ( !*(_DWORD *)(*((_QWORD *)v85 - 1) + 40LL) )
        goto LABEL_31;
      v88 = 0;
      v10 = 0;
      v86 = *((_QWORD *)v85 - 1);
      while ( 1 )
      {
        i = *(_QWORD *)(*(_QWORD *)(v86 + 48) + 8LL * v88);
        if ( !*(_DWORD *)(i + 24) )
        {
          v15 = *(_QWORD *)i;
          v16 = *a1;
          v17 = *(_QWORD *)(*a1 + 32);
          v89 = *(_QWORD *)(*a1 + 40);
          v18 = (__int64 *)sub_2E311E0(*(_QWORD *)i);
          sub_356E080(0xAu, v15, v18, *(_QWORD *)(v16 + 8), *(_QWORD *)(v16 + 16), v89, v17);
          *(_DWORD *)(i + 8) = *(_DWORD *)(*(_QWORD *)(v19 + 32) + 8LL);
          v20 = a1[1];
          v21 = *(_DWORD *)(v20 + 24);
          if ( v21 )
          {
            v22 = *(_QWORD *)(v20 + 8);
            v23 = 1;
            v24 = 0;
            v25 = (v21 - 1) & (((unsigned int)*(_QWORD *)i >> 9) ^ ((unsigned int)*(_QWORD *)i >> 4));
            v26 = (_QWORD *)(v22 + 16LL * v25);
            v27 = *v26;
            if ( *v26 == *(_QWORD *)i )
            {
LABEL_26:
              v28 = v26 + 1;
LABEL_27:
              *v28 = *(_DWORD *)(i + 8);
              *(_QWORD *)(i + 16) = i;
              v29 = *(_DWORD *)(v84 + 24);
              *(_DWORD *)(i + 24) = v29;
              *(_DWORD *)(v84 + 24) = v29 + 1;
              goto LABEL_11;
            }
            while ( v27 != -4096 )
            {
              if ( v27 == -8192 && !v24 )
                v24 = v26;
              v25 = (v21 - 1) & (v23 + v25);
              v26 = (_QWORD *)(v22 + 16LL * v25);
              v27 = *v26;
              if ( *(_QWORD *)i == *v26 )
                goto LABEL_26;
              ++v23;
            }
            v43 = *(_DWORD *)(v20 + 16);
            if ( !v24 )
              v24 = v26;
            ++*(_QWORD *)v20;
            v44 = v43 + 1;
            if ( 4 * (v43 + 1) < 3 * v21 )
            {
              if ( v21 - *(_DWORD *)(v20 + 20) - v44 > v21 >> 3 )
                goto LABEL_58;
              sub_34F9190(v20, v21);
              v53 = *(_DWORD *)(v20 + 24);
              if ( !v53 )
              {
LABEL_126:
                ++*(_DWORD *)(v20 + 16);
                BUG();
              }
              v54 = v53 - 1;
              v55 = *(_QWORD *)(v20 + 8);
              v56 = 1;
              v44 = *(_DWORD *)(v20 + 16) + 1;
              v52 = 0;
              LODWORD(v57) = v54 & (((unsigned int)*(_QWORD *)i >> 9) ^ ((unsigned int)*(_QWORD *)i >> 4));
              v24 = (_QWORD *)(v55 + 16LL * (unsigned int)v57);
              v58 = *v24;
              if ( *(_QWORD *)i == *v24 )
                goto LABEL_58;
              while ( v58 != -4096 )
              {
                if ( v58 == -8192 && !v52 )
                  v52 = v24;
                v57 = v54 & (unsigned int)(v57 + v56);
                v24 = (_QWORD *)(v55 + 16 * v57);
                v58 = *v24;
                if ( *(_QWORD *)i == *v24 )
                  goto LABEL_58;
                ++v56;
              }
              goto LABEL_74;
            }
          }
          else
          {
            ++*(_QWORD *)v20;
          }
          sub_34F9190(v20, 2 * v21);
          v46 = *(_DWORD *)(v20 + 24);
          if ( !v46 )
            goto LABEL_126;
          v47 = v46 - 1;
          v48 = *(_QWORD *)(v20 + 8);
          v44 = *(_DWORD *)(v20 + 16) + 1;
          LODWORD(v49) = v47 & (((unsigned int)*(_QWORD *)i >> 9) ^ ((unsigned int)*(_QWORD *)i >> 4));
          v24 = (_QWORD *)(v48 + 16LL * (unsigned int)v49);
          v50 = *v24;
          if ( *(_QWORD *)i == *v24 )
            goto LABEL_58;
          v51 = 1;
          v52 = 0;
          while ( v50 != -4096 )
          {
            if ( v50 == -8192 && !v52 )
              v52 = v24;
            v49 = v47 & (unsigned int)(v49 + v51);
            v24 = (_QWORD *)(v48 + 16 * v49);
            v50 = *v24;
            if ( *(_QWORD *)i == *v24 )
              goto LABEL_58;
            ++v51;
          }
LABEL_74:
          if ( v52 )
            v24 = v52;
LABEL_58:
          *(_DWORD *)(v20 + 16) = v44;
          if ( *v24 != -4096 )
            --*(_DWORD *)(v20 + 20);
          v45 = *(_QWORD *)i;
          v28 = v24 + 1;
          *v28 = 0;
          *((_QWORD *)v28 - 1) = v45;
          goto LABEL_27;
        }
LABEL_11:
        if ( v10 && i != v10 )
        {
          v12 = *(_DWORD *)(i + 24);
          v13 = i;
          v14 = *(_DWORD *)(v10 + 24);
          for ( i = v10; ; v14 = *(_DWORD *)(i + 24) )
          {
            while ( v14 >= v12 )
            {
              while ( v14 > v12 )
              {
                v13 = *(_QWORD *)(v13 + 32);
                if ( !v13 )
                  goto LABEL_22;
                v12 = *(_DWORD *)(v13 + 24);
              }
              if ( i == v13 )
                goto LABEL_22;
            }
            i = *(_QWORD *)(i + 32);
            if ( !i )
              break;
          }
          i = v13;
        }
LABEL_22:
        if ( *(_DWORD *)(v86 + 40) == ++v88 )
          break;
        v10 = i;
      }
      if ( i && *(_QWORD *)(v86 + 32) != i )
      {
        *(_QWORD *)(v86 + 32) = i;
        v87 = 1;
      }
LABEL_31:
      v85 -= 8;
    }
    while ( (_BYTE *)v83 != v85 );
    v2 = (unsigned int)v93;
  }
  while ( v87 );
  v30 = (unsigned __int64)v92;
  v31 = (__int64)a1;
  v32 = &v92[8 * (unsigned int)v93];
  do
  {
    if ( (_BYTE *)v30 == v32 )
      break;
    v33 = 0;
    v34 = v32;
    do
    {
      v35 = *((_QWORD *)v34 - 1);
      v36 = *(_QWORD *)(v35 + 16);
      if ( v35 == v36 )
        goto LABEL_45;
      v37 = *(_DWORD *)(v35 + 40);
      v38 = *(_QWORD *)(v35 + 32);
      if ( v37 )
      {
        v39 = *(__int64 **)(v35 + 48);
        v40 = (__int64)&v39[(unsigned int)(v37 - 1) + 1];
        while ( 1 )
        {
          v41 = *v39;
          if ( v38 != *v39 )
            break;
LABEL_83:
          if ( (__int64 *)v40 == ++v39 )
            goto LABEL_84;
        }
        while ( *(_QWORD *)(v41 + 16) != v41 )
        {
          v41 = *(_QWORD *)(v41 + 32);
          if ( v38 == v41 )
            goto LABEL_83;
        }
        v42 = *((_QWORD *)v34 - 1);
LABEL_44:
        *(_QWORD *)(v35 + 16) = v42;
        v33 = 1;
        goto LABEL_45;
      }
LABEL_84:
      v42 = *(_QWORD *)(v38 + 16);
      if ( v36 != v42 )
        goto LABEL_44;
LABEL_45:
      v34 -= 8;
    }
    while ( (_BYTE *)v30 != v34 );
  }
  while ( v33 );
LABEL_79:
  sub_3570360(v31, (__int64)&v92);
  v59 = *(_DWORD *)(v31 + 48);
  if ( v59 )
  {
    v60 = v91;
    v61 = *(_QWORD *)(v31 + 32);
    v62 = 0;
    v63 = 1;
    v64 = (v59 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
    v65 = (__int64 *)(v61 + 16LL * v64);
    v66 = *v65;
    if ( v91 == *v65 )
    {
LABEL_81:
      v67 = v65[1];
      goto LABEL_82;
    }
    while ( v66 != -4096 )
    {
      if ( v66 == -8192 && !v62 )
        v62 = v65;
      v64 = (v59 - 1) & (v63 + v64);
      v65 = (__int64 *)(v61 + 16LL * v64);
      v66 = *v65;
      if ( v91 == *v65 )
        goto LABEL_81;
      ++v63;
    }
    if ( v62 )
      v65 = v62;
    v76 = *(_DWORD *)(v31 + 40);
    ++*(_QWORD *)(v31 + 24);
    v71 = v76 + 1;
    if ( 4 * (v76 + 1) < 3 * v59 )
    {
      if ( v59 - *(_DWORD *)(v31 + 44) - v71 > v59 >> 3 )
        goto LABEL_89;
      sub_356EA90(v31 + 24, v59);
      v77 = *(_DWORD *)(v31 + 48);
      if ( v77 )
      {
        v78 = v77 - 1;
        v79 = *(_QWORD *)(v31 + 32);
        v75 = 0;
        v60 = v91;
        v71 = *(_DWORD *)(v31 + 40) + 1;
        v80 = 1;
        v81 = v78 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
        v65 = (__int64 *)(v79 + 16LL * v81);
        v82 = *v65;
        if ( *v65 != v91 )
        {
          while ( v82 != -4096 )
          {
            if ( !v75 && v82 == -8192 )
              v75 = v65;
            v81 = v78 & (v80 + v81);
            v65 = (__int64 *)(v79 + 16LL * v81);
            v82 = *v65;
            if ( v91 == *v65 )
              goto LABEL_89;
            ++v80;
          }
          goto LABEL_94;
        }
        goto LABEL_89;
      }
LABEL_125:
      ++*(_DWORD *)(v31 + 40);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v31 + 24);
  }
  sub_356EA90(v31 + 24, 2 * v59);
  v68 = *(_DWORD *)(v31 + 48);
  if ( !v68 )
    goto LABEL_125;
  v60 = v91;
  v69 = v68 - 1;
  v70 = *(_QWORD *)(v31 + 32);
  v71 = *(_DWORD *)(v31 + 40) + 1;
  v72 = v69 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
  v65 = (__int64 *)(v70 + 16LL * v72);
  v73 = *v65;
  if ( *v65 != v91 )
  {
    v74 = 1;
    v75 = 0;
    while ( v73 != -4096 )
    {
      if ( !v75 && v73 == -8192 )
        v75 = v65;
      v72 = v69 & (v74 + v72);
      v65 = (__int64 *)(v70 + 16LL * v72);
      v73 = *v65;
      if ( v91 == *v65 )
        goto LABEL_89;
      ++v74;
    }
LABEL_94:
    if ( v75 )
      v65 = v75;
  }
LABEL_89:
  *(_DWORD *)(v31 + 40) = v71;
  if ( *v65 != -4096 )
    --*(_DWORD *)(v31 + 44);
  *v65 = v60;
  v65[1] = 0;
  v67 = 0;
LABEL_82:
  v8 = *(_DWORD *)(*(_QWORD *)(v67 + 16) + 8LL);
LABEL_3:
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  return v8;
}
