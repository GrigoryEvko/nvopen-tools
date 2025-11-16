// Function: sub_C19C00
// Address: 0xc19c00
//
_BYTE *__fastcall sub_C19C00(__int64 a1, __int64 a2, __int64 a3, char *a4)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned int v7; // eax
  __int64 *v8; // rbx
  __int64 *v9; // r12
  __int64 *v10; // rdi
  _BYTE *result; // rax
  __int64 *v12; // r15
  signed __int64 v13; // rcx
  char **v14; // r15
  __int64 *v15; // r13
  __int64 *v16; // r8
  __int64 *v17; // rdx
  int v18; // r12d
  __int64 *v19; // rbx
  char *v20; // rdx
  int v21; // eax
  unsigned __int64 v22; // r12
  __int64 v23; // r13
  unsigned __int64 v24; // rax
  _QWORD *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 *v31; // r12
  __int64 v32; // r14
  __int64 *v33; // r13
  __int64 v34; // r9
  unsigned __int64 v35; // r15
  unsigned int v36; // r8d
  _QWORD *v37; // rax
  __int64 v38; // rdi
  __int64 *v39; // rdx
  int v40; // eax
  signed __int64 v41; // rdx
  int v42; // r11d
  int v43; // r15d
  int v44; // r15d
  __int64 v45; // r10
  int v46; // eax
  _QWORD *v47; // rcx
  __int64 v48; // r9
  int v49; // r8d
  _QWORD *v50; // rdi
  __int64 v51; // rcx
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rdx
  int *v54; // rcx
  int v55; // edi
  __int64 *v56; // rbx
  __int64 *v57; // rdi
  int v58; // eax
  _QWORD *v59; // rcx
  _QWORD *v60; // rdx
  _QWORD *v61; // rax
  int v62; // eax
  int v63; // r9d
  __int64 v64; // r10
  unsigned int v65; // r15d
  int v66; // edi
  __int64 v67; // r8
  int v68; // eax
  __int64 v69; // r15
  __int64 v70; // r13
  __int64 v71; // rdi
  signed __int64 v72; // [rsp+0h] [rbp-60h]
  char *v73; // [rsp+8h] [rbp-58h]
  __int64 v74; // [rsp+10h] [rbp-50h]
  unsigned __int64 v75; // [rsp+18h] [rbp-48h]
  __int64 *v76; // [rsp+20h] [rbp-40h] BYREF
  __int64 v77; // [rsp+28h] [rbp-38h]
  _BYTE v78[48]; // [rsp+30h] [rbp-30h] BYREF

  v5 = a2;
  v6 = a1;
  v74 = a3;
  sub_C177B0(a2);
  v77 = 0;
  v76 = (__int64 *)v78;
  v7 = *(_DWORD *)(a2 + 40);
  if ( !v7 )
    goto LABEL_2;
  v12 = *(__int64 **)(a2 + 32);
  v13 = v7;
  if ( v12 != (__int64 *)(a2 + 48) )
  {
    LODWORD(v77) = *(_DWORD *)(a2 + 40);
    v21 = *(_DWORD *)(a2 + 44);
    v76 = v12;
    HIDWORD(v77) = v21;
    *(_QWORD *)(a2 + 32) = a2 + 48;
    *(_QWORD *)(a2 + 40) = 0;
    goto LABEL_20;
  }
  LODWORD(v73) = *(_DWORD *)(a2 + 40);
  v75 = v7;
  sub_C18000((__int64)&v76, v7);
  v14 = *(char ***)(a2 + 32);
  v15 = v76;
  v13 = v75;
  a2 = 9LL * *(unsigned int *)(a2 + 40);
  v16 = (__int64 *)&v14[9 * *(unsigned int *)(v5 + 40)];
  v17 = v76;
  if ( v14 == (char **)v16 )
  {
    LODWORD(v77) = (_DWORD)v73;
  }
  else
  {
    v75 = v5;
    v18 = (int)v73;
    v73 = a4;
    v19 = v16;
    do
    {
      if ( v15 )
      {
        v20 = *v14;
        *((_DWORD *)v15 + 4) = 0;
        *((_DWORD *)v15 + 5) = 12;
        *v15 = (__int64)v20;
        v15[1] = (__int64)(v15 + 3);
        if ( *((_DWORD *)v14 + 4) )
        {
          a2 = (__int64)(v14 + 1);
          v72 = v13;
          sub_C15E20((__int64)(v15 + 1), v14 + 1);
          v13 = v72;
        }
      }
      v14 += 9;
      v15 += 9;
    }
    while ( v19 != (__int64 *)v14 );
    v68 = v18;
    v5 = v75;
    v6 = a1;
    a4 = v73;
    LODWORD(v77) = v68;
    v69 = *(_QWORD *)(v75 + 32);
    v70 = v69 + 72LL * *(unsigned int *)(v75 + 40);
    if ( v70 != v69 )
    {
      do
      {
        v70 -= 72;
        v71 = *(_QWORD *)(v70 + 8);
        if ( v71 != v70 + 24 )
          _libc_free(v71, a2);
      }
      while ( v70 != v69 );
      *(_DWORD *)(v75 + 40) = 0;
      if ( !(_DWORD)v77 )
      {
LABEL_2:
        if ( *(_QWORD *)v6 != *(_QWORD *)(v6 + 8) )
          *(_QWORD *)(v6 + 8) = *(_QWORD *)v6;
        sub_C177B0(v6 + 24);
        v8 = v76;
        v9 = &v76[9 * (unsigned int)v77];
        if ( v76 != v9 )
        {
          do
          {
            v9 -= 9;
            v10 = (__int64 *)v9[1];
            if ( v10 != v9 + 3 )
              _libc_free(v10, a2);
          }
          while ( v8 != v9 );
LABEL_8:
          v9 = v76;
          goto LABEL_9;
        }
        goto LABEL_9;
      }
      v17 = v76;
      v13 = (unsigned int)v77;
      goto LABEL_93;
    }
    v17 = v76;
  }
  *(_DWORD *)(v5 + 40) = 0;
LABEL_93:
  v12 = v17;
LABEL_20:
  v22 = 9 * v13;
  v23 = (__int64)&v12[9 * v13];
  _BitScanReverse64(&v24, 0x8E38E38E38E38E39LL * ((72 * v13) >> 3));
  sub_C198C0((__int64)v12, v23, 2LL * (int)(63 - (v24 ^ 0x3F)), a4);
  if ( v22 <= 144 )
  {
    sub_C19340(v12, &v12[v22], (__int64)a4);
  }
  else
  {
    v25 = v12 + 144;
    sub_C19340(v12, v12 + 144, (__int64)a4);
    if ( (__int64 *)v23 != v12 + 144 )
    {
      do
      {
        v26 = (__int64)v25;
        v25 += 9;
        sub_C19270(v26, (__int64)a4);
      }
      while ( (_QWORD *)v23 != v25 );
    }
  }
  if ( *(_QWORD *)v6 != *(_QWORD *)(v6 + 8) )
    *(_QWORD *)(v6 + 8) = *(_QWORD *)v6;
  sub_C17980(v6, 8LL * (unsigned int)v77);
  v27 = *(_QWORD *)(v6 + 56);
  if ( v27 != *(_QWORD *)(v6 + 64) )
    *(_QWORD *)(v6 + 64) = v27;
  a2 = 512;
  sub_C17980(v6 + 56, 0x200u);
  v73 = (char *)(v6 + 24);
  sub_C177B0(v6 + 24);
  v28 = (unsigned int)v77;
  if ( (_DWORD)v77 )
  {
    ++*(_QWORD *)(v6 + 24);
    v29 = ((((((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v28 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v28 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 8;
    v30 = (((v29
           | (((((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v28 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v28 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 16)
         | v29
         | (((((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v28 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v28 / 3u + 1) | ((unsigned __int64)(4 * (int)v28 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v28 / 3u + 1)
         | ((4 * (int)v28 / 3u + 1) >> 1))
        + 1;
    if ( *(_DWORD *)(v6 + 48) < v30 )
    {
      sub_9E25D0(v6 + 24, v30);
      v28 = (unsigned int)v77;
    }
    a2 = (__int64)v76;
    v31 = &v76[9 * v28];
    v75 = (unsigned __int64)v76;
    if ( v76 != v31 )
    {
      v32 = v74;
      v33 = 0;
      while ( 1 )
      {
        v39 = v33;
        v33 = v31 - 8;
        v40 = sub_C17C10((_QWORD *)v6, v31 - 8, v39, v32);
        a2 = *(unsigned int *)(v6 + 48);
        v41 = *(v31 - 9);
        v42 = v40;
        if ( !(_DWORD)a2 )
          break;
        v34 = *(_QWORD *)(v6 + 32);
        v35 = ((0xBF58476D1CE4E5B9LL * v41) >> 31) ^ (0xBF58476D1CE4E5B9LL * v41);
        v36 = v35 & (a2 - 1);
        v37 = (_QWORD *)(v34 + 16LL * v36);
        v38 = *v37;
        if ( v41 != *v37 )
        {
          LODWORD(v74) = 1;
          v47 = 0;
          while ( v38 != -1 )
          {
            if ( v47 || v38 != -2 )
              v37 = v47;
            v36 = (a2 - 1) & (v74 + v36);
            v72 = v34 + 16LL * v36;
            v38 = *(_QWORD *)v72;
            if ( v41 == *(_QWORD *)v72 )
              goto LABEL_33;
            LODWORD(v74) = v74 + 1;
            v47 = v37;
            v37 = (_QWORD *)v72;
          }
          if ( !v47 )
            v47 = v37;
          v58 = *(_DWORD *)(v6 + 40);
          ++*(_QWORD *)(v6 + 24);
          v46 = v58 + 1;
          if ( 4 * v46 < (unsigned int)(3 * a2) )
          {
            if ( (int)a2 - *(_DWORD *)(v6 + 44) - v46 <= (unsigned int)a2 >> 3 )
            {
              v72 = v41;
              LODWORD(v74) = v42;
              sub_9E25D0((__int64)v73, a2);
              v62 = *(_DWORD *)(v6 + 48);
              if ( !v62 )
              {
LABEL_106:
                ++*(_DWORD *)(v6 + 40);
                BUG();
              }
              v63 = v62 - 1;
              v64 = *(_QWORD *)(v6 + 32);
              a2 = 0;
              v65 = (v62 - 1) & v35;
              v41 = v72;
              v42 = v74;
              v66 = 1;
              v46 = *(_DWORD *)(v6 + 40) + 1;
              v47 = (_QWORD *)(v64 + 16LL * v65);
              v67 = *v47;
              if ( v72 != *v47 )
              {
                while ( v67 != -1 )
                {
                  if ( !a2 && v67 == -2 )
                    a2 = (__int64)v47;
                  v65 = v63 & (v66 + v65);
                  LODWORD(v74) = v66 + 1;
                  v47 = (_QWORD *)(v64 + 16LL * v65);
                  v67 = *v47;
                  if ( v72 == *v47 )
                    goto LABEL_60;
                  v66 = v74;
                }
                if ( a2 )
                  v47 = (_QWORD *)a2;
              }
            }
            goto LABEL_60;
          }
LABEL_36:
          v72 = v41;
          LODWORD(v74) = v42;
          sub_9E25D0((__int64)v73, 2 * a2);
          v43 = *(_DWORD *)(v6 + 48);
          if ( !v43 )
            goto LABEL_106;
          v41 = v72;
          v44 = v43 - 1;
          v45 = *(_QWORD *)(v6 + 32);
          v42 = v74;
          a2 = v44 & ((unsigned int)((0xBF58476D1CE4E5B9LL * v72) >> 31) ^ (484763065 * (_DWORD)v72));
          v46 = *(_DWORD *)(v6 + 40) + 1;
          v47 = (_QWORD *)(v45 + 16 * a2);
          v48 = *v47;
          if ( v72 != *v47 )
          {
            v49 = 1;
            v50 = 0;
            while ( v48 != -1 )
            {
              if ( v48 == -2 && !v50 )
                v50 = v47;
              a2 = v44 & (unsigned int)(v49 + a2);
              LODWORD(v74) = v49 + 1;
              v47 = (_QWORD *)(v45 + 16LL * (unsigned int)a2);
              v48 = *v47;
              if ( v72 == *v47 )
                goto LABEL_60;
              v49 = v74;
            }
            if ( v50 )
              v47 = v50;
          }
LABEL_60:
          *(_DWORD *)(v6 + 40) = v46;
          if ( *v47 != -1 )
            --*(_DWORD *)(v6 + 44);
          *v47 = v41;
          *((_DWORD *)v47 + 2) = v42;
        }
LABEL_33:
        v31 -= 9;
        if ( (__int64 *)v75 == v31 )
          goto LABEL_44;
      }
      ++*(_QWORD *)(v6 + 24);
      goto LABEL_36;
    }
  }
  else
  {
    ++*(_QWORD *)(v6 + 24);
  }
LABEL_44:
  v51 = *(_QWORD *)v6;
  v52 = ((__int64)(*(_QWORD *)(v6 + 8) - *(_QWORD *)v6) >> 2) - 1;
  if ( (__int64)(*(_QWORD *)(v6 + 8) - *(_QWORD *)v6) >> 2 != 1 )
  {
    v53 = 0;
    while ( 1 )
    {
      a2 = v51 + 4 * v52;
      v54 = (int *)(v51 + 4 * v53);
      --v52;
      ++v53;
      v55 = *v54;
      *v54 = *(_DWORD *)a2;
      *(_DWORD *)a2 = v55;
      if ( v53 >= v52 )
        break;
      v51 = *(_QWORD *)v6;
    }
  }
  if ( *(_DWORD *)(v6 + 40) )
  {
    v59 = *(_QWORD **)(v6 + 32);
    v60 = &v59[2 * *(unsigned int *)(v6 + 48)];
    if ( v59 != v60 )
    {
      while ( 1 )
      {
        v61 = v59;
        if ( *v59 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v59 += 2;
        if ( v60 == v59 )
          goto LABEL_49;
      }
      if ( v60 != v59 )
      {
        do
        {
          v61 += 2;
          *((_DWORD *)v61 - 2) = ((__int64)(*(_QWORD *)(v6 + 8) - *(_QWORD *)v6) >> 2) - 1 - *((_DWORD *)v61 - 2);
          if ( v61 == v60 )
            break;
          while ( *v61 > 0xFFFFFFFFFFFFFFFDLL )
          {
            v61 += 2;
            if ( v60 == v61 )
              goto LABEL_49;
          }
        }
        while ( v61 != v60 );
      }
    }
  }
LABEL_49:
  v56 = v76;
  v9 = &v76[9 * (unsigned int)v77];
  if ( v76 != v9 )
  {
    do
    {
      v9 -= 9;
      v57 = (__int64 *)v9[1];
      if ( v57 != v9 + 3 )
        _libc_free(v57, a2);
    }
    while ( v56 != v9 );
    goto LABEL_8;
  }
LABEL_9:
  result = v78;
  if ( v9 != (__int64 *)v78 )
    return (_BYTE *)_libc_free(v9, a2);
  return result;
}
