// Function: sub_27320E0
// Address: 0x27320e0
//
__int64 __fastcall sub_27320E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 *v12; // rbx
  __int64 *v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // r15
  __int64 v22; // rdx
  __int64 *v23; // r14
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r15
  __int64 v27; // r13
  unsigned __int64 v28; // rdi
  unsigned __int64 *v29; // r12
  unsigned __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r12
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  unsigned __int64 *v36; // rbx
  __int64 v37; // rax
  __int64 *v38; // r13
  __int64 v39; // rdx
  __int64 v40; // r14
  unsigned __int64 v41; // r12
  __int64 v42; // rcx
  __int64 v43; // rbx
  __int64 v44; // rcx
  __int64 v45; // rsi
  unsigned __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // r14
  unsigned __int64 *v49; // r15
  __int64 v50; // rbx
  __int64 v51; // r13
  char *v52; // r13
  __int64 v53; // rax
  unsigned __int64 v54; // r12
  unsigned __int64 v55; // rbx
  __int64 v56; // rax
  unsigned __int64 v57; // r15
  unsigned __int64 *v58; // r14
  __int64 result; // rax
  unsigned int v60; // ecx
  unsigned int v61; // eax
  int v62; // ebx
  __int64 v63; // rax
  __int64 v64; // rbx
  unsigned int v65; // kr00_4
  __int64 v66; // rbx
  unsigned __int64 v67; // rdx
  unsigned __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rdx
  _QWORD *j; // rdx
  __int64 v72; // [rsp+8h] [rbp-1578h]
  unsigned int v73; // [rsp+14h] [rbp-156Ch]
  __int64 *v74; // [rsp+18h] [rbp-1568h]
  __int64 v75; // [rsp+20h] [rbp-1560h]
  char *v76; // [rsp+28h] [rbp-1558h]
  __int64 v77; // [rsp+30h] [rbp-1550h]
  char *v78; // [rsp+38h] [rbp-1548h] BYREF
  __int64 v79; // [rsp+40h] [rbp-1540h]
  char v80; // [rsp+48h] [rbp-1538h] BYREF

  v6 = *(_DWORD *)(a1 + 5592);
  ++*(_QWORD *)(a1 + 5576);
  v75 = a1;
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 5596) )
      goto LABEL_7;
    v7 = *(unsigned int *)(a1 + 5600);
    if ( (unsigned int)v7 > 0x40 )
    {
      v64 = v75;
      a2 = 16LL * *(unsigned int *)(a1 + 5600);
      a1 = *(_QWORD *)(v75 + 5584);
      sub_C7D6A0(a1, a2, 8);
      *(_QWORD *)(v75 + 5584) = 0;
      *(_QWORD *)(v64 + 5592) = 0;
      *(_DWORD *)(v64 + 5600) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a2 = 64;
  v7 = *(unsigned int *)(v75 + 5600);
  v60 = 4 * v6;
  if ( (unsigned int)(4 * v6) < 0x40 )
    v60 = 64;
  if ( v60 >= (unsigned int)v7 )
  {
LABEL_4:
    v8 = *(_QWORD **)(v75 + 5584);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(v75 + 5592) = 0;
    goto LABEL_7;
  }
  v61 = v6 - 1;
  if ( v61 )
  {
    _BitScanReverse(&v61, v61);
    v62 = 1 << (33 - (v61 ^ 0x1F));
    a1 = *(_QWORD *)(v75 + 5584);
    if ( v62 < 64 )
      v62 = 64;
    if ( v62 == (_DWORD)v7 )
    {
      *(_QWORD *)(v75 + 5592) = 0;
      v63 = a1 + 16LL * (unsigned int)v62;
      do
      {
        if ( a1 )
          *(_QWORD *)a1 = -4096;
        a1 += 16;
      }
      while ( v63 != a1 );
      goto LABEL_7;
    }
  }
  else
  {
    v62 = 64;
    a1 = *(_QWORD *)(v75 + 5584);
  }
  sub_C7D6A0(a1, 16LL * *(unsigned int *)(v75 + 5600), 8);
  a2 = 8;
  v65 = 4 * v62;
  v66 = v75;
  v67 = ((((((((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
           | (v65 / 3 + 1)
           | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 4)
         | (((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
         | (v65 / 3 + 1)
         | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 8)
       | (((((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
         | (v65 / 3 + 1)
         | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 4)
       | (((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
       | (v65 / 3 + 1)
       | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 16;
  v68 = (v67
       | (((((((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
           | (v65 / 3 + 1)
           | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 4)
         | (((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
         | (v65 / 3 + 1)
         | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 8)
       | (((((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
         | (v65 / 3 + 1)
         | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 4)
       | (((v65 / 3 + 1) | ((unsigned __int64)(v65 / 3 + 1) >> 1)) >> 2)
       | (v65 / 3 + 1)
       | ((unsigned __int64)(v65 / 3 + 1) >> 1))
      + 1;
  *(_DWORD *)(v75 + 5600) = v68;
  a1 = 16 * v68;
  v69 = (_QWORD *)sub_C7D670(16 * v68, 8);
  v70 = *(unsigned int *)(v66 + 5600);
  *(_QWORD *)(v66 + 5592) = 0;
  *(_QWORD *)(v66 + 5584) = v69;
  for ( j = &v69[2 * v70]; j != v69; v69 += 2 )
  {
    if ( v69 )
      *v69 = -4096;
  }
LABEL_7:
  v10 = *(__int64 **)(v75 + 64);
  v11 = *(__int64 **)(v75 + 72);
  *(_DWORD *)(v75 + 5616) = 0;
  if ( v10 != v11 )
  {
    v12 = v10;
    do
    {
      a1 = *v12;
      if ( (__int64 *)*v12 != v12 + 2 )
        _libc_free(a1);
      v12 += 21;
    }
    while ( v11 != v12 );
    *(_QWORD *)(v75 + 72) = v10;
  }
  v13 = *(__int64 **)(v75 + 120);
  v14 = 4LL * *(unsigned int *)(v75 + 128);
  v74 = &v13[v14];
  while ( v13 != v74 )
  {
    v15 = v13[2];
    v16 = v13[1];
    v17 = v15 - v16;
    if ( v15 == v16 )
    {
      v19 = 0;
      if ( v15 == v16 )
        goto LABEL_29;
    }
    else
    {
      if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(a1, a2, v16);
      a1 = v13[2] - v16;
      v18 = sub_22077B0(v17);
      v15 = v13[2];
      v16 = v13[1];
      v19 = v18;
      if ( v15 == v16 )
        goto LABEL_28;
    }
    v20 = v16;
    v21 = v19;
    do
    {
      if ( v21 )
      {
        *(_DWORD *)(v21 + 8) = 0;
        *(_QWORD *)v21 = v21 + 16;
        *(_DWORD *)(v21 + 12) = 8;
        v22 = *(unsigned int *)(v20 + 8);
        if ( (_DWORD)v22 )
        {
          v76 = (char *)v15;
          sub_272D7C0(v21, v20, v22, v15, a5, a6);
          v15 = (__int64)v76;
        }
        *(_QWORD *)(v21 + 144) = *(_QWORD *)(v20 + 144);
        *(_QWORD *)(v21 + 152) = *(_QWORD *)(v20 + 152);
        a2 = *(unsigned int *)(v20 + 160);
        *(_DWORD *)(v21 + 160) = a2;
      }
      v20 += 168;
      v21 += 168;
    }
    while ( v15 != v20 );
    if ( v19 == v21 )
    {
LABEL_88:
      a2 = v17;
      a1 = v19;
      j_j___libc_free_0(v19);
      goto LABEL_29;
    }
    v23 = (__int64 *)v19;
    do
    {
      a1 = *v23;
      if ( (__int64 *)*v23 != v23 + 2 )
        _libc_free(a1);
      v23 += 21;
    }
    while ( (__int64 *)v21 != v23 );
LABEL_28:
    if ( v19 )
      goto LABEL_88;
LABEL_29:
    v13 += 4;
  }
  sub_2730D40(v75 + 88);
  v26 = *(_QWORD *)(v75 + 120);
  v27 = v26 + 32LL * *(unsigned int *)(v75 + 128);
  while ( v26 != v27 )
  {
    v28 = *(_QWORD *)(v27 - 24);
    v29 = *(unsigned __int64 **)(v27 - 16);
    v27 -= 32;
    v30 = (unsigned __int64 *)v28;
    if ( v29 != (unsigned __int64 *)v28 )
    {
      do
      {
        if ( (unsigned __int64 *)*v30 != v30 + 2 )
          _libc_free(*v30);
        v30 += 21;
      }
      while ( v29 != v30 );
      v28 = *(_QWORD *)(v27 + 8);
    }
    if ( v28 )
      j_j___libc_free_0(v28);
  }
  v31 = v75;
  v32 = *(_QWORD *)(v75 + 136);
  *(_DWORD *)(v75 + 128) = 0;
  v33 = v32 + 672LL * *(unsigned int *)(v31 + 144);
  while ( v32 != v33 )
  {
    v34 = *(unsigned int *)(v33 - 648);
    v35 = *(_QWORD *)(v33 - 656);
    v33 -= 672;
    v36 = (unsigned __int64 *)(v35 + 160 * v34);
    if ( (unsigned __int64 *)v35 != v36 )
    {
      do
      {
        v36 -= 20;
        if ( (unsigned __int64 *)*v36 != v36 + 2 )
          _libc_free(*v36);
      }
      while ( (unsigned __int64 *)v35 != v36 );
      v35 = *(_QWORD *)(v33 + 16);
    }
    if ( v35 != v33 + 32 )
      _libc_free(v35);
  }
  v37 = v75;
  v38 = *(__int64 **)(v75 + 5560);
  *(_DWORD *)(v75 + 144) = 0;
  v74 = &v38[675 * *(unsigned int *)(v37 + 5568)];
  if ( v74 != v38 )
  {
    v76 = &v80;
    do
    {
      while ( 1 )
      {
        v77 = *v38;
        v78 = v76;
        v79 = 0x800000000LL;
        v39 = *((unsigned int *)v38 + 4);
        if ( (_DWORD)v39 )
        {
          if ( &v78 != (char **)(v38 + 1) )
          {
            v40 = (unsigned int)v39;
            v41 = (unsigned __int64)v76;
            v42 = (unsigned int)v39;
            if ( (unsigned int)v39 > 8 )
            {
              v73 = *((_DWORD *)v38 + 4);
              sub_23672E0((__int64)&v78, (unsigned int)v39, v39, (unsigned int)v39, v24, v25);
              v41 = (unsigned __int64)v78;
              v42 = *((unsigned int *)v38 + 4);
              v39 = v73;
            }
            v43 = v38[1];
            v44 = v43 + 672 * v42;
            if ( v43 != v44 )
            {
              do
              {
                if ( v41 )
                {
                  *(_QWORD *)v41 = *(_QWORD *)v43;
                  v45 = *(_QWORD *)(v43 + 8);
                  *(_DWORD *)(v41 + 24) = 0;
                  *(_QWORD *)(v41 + 8) = v45;
                  *(_QWORD *)(v41 + 16) = v41 + 32;
                  *(_DWORD *)(v41 + 28) = 4;
                  if ( *(_DWORD *)(v43 + 24) )
                  {
                    v72 = v44;
                    v73 = v39;
                    sub_2731610(v41 + 16, v43 + 16, v39, v44, v24, v25);
                    v44 = v72;
                    v39 = v73;
                  }
                }
                v43 += 672;
                v41 += 672LL;
              }
              while ( v44 != v43 );
              v41 = (unsigned __int64)v78;
            }
            LODWORD(v79) = v39;
            v46 = v41 + 672 * v40;
            do
            {
              v47 = *(unsigned int *)(v46 - 648);
              v48 = *(_QWORD *)(v46 - 656);
              v46 -= 672LL;
              v47 *= 160;
              v49 = (unsigned __int64 *)(v48 + v47);
              if ( v48 != v48 + v47 )
              {
                do
                {
                  v49 -= 20;
                  if ( (unsigned __int64 *)*v49 != v49 + 2 )
                    _libc_free(*v49);
                }
                while ( (unsigned __int64 *)v48 != v49 );
                v48 = *(_QWORD *)(v46 + 16);
              }
              if ( v48 != v46 + 32 )
                _libc_free(v48);
            }
            while ( v46 != v41 );
            if ( v78 != v76 )
              break;
          }
        }
        v38 += 675;
        if ( v74 == v38 )
          goto LABEL_73;
      }
      _libc_free((unsigned __int64)v78);
      v38 += 675;
    }
    while ( v74 != v38 );
  }
LABEL_73:
  v50 = v75;
  sub_2730D40(v75 + 5528);
  v51 = 5400LL * *(unsigned int *)(v50 + 5568);
  v76 = *(char **)(v50 + 5560);
  v52 = &v76[v51];
  while ( v76 != v52 )
  {
    v53 = *((unsigned int *)v52 - 1346);
    v54 = *((_QWORD *)v52 - 674);
    v52 -= 5400;
    v55 = v54 + 672 * v53;
    if ( v54 != v55 )
    {
      do
      {
        v56 = *(unsigned int *)(v55 - 648);
        v57 = *(_QWORD *)(v55 - 656);
        v55 -= 672LL;
        v56 *= 160;
        v58 = (unsigned __int64 *)(v57 + v56);
        if ( v57 != v57 + v56 )
        {
          do
          {
            v58 -= 20;
            if ( (unsigned __int64 *)*v58 != v58 + 2 )
              _libc_free(*v58);
          }
          while ( (unsigned __int64 *)v57 != v58 );
          v57 = *(_QWORD *)(v55 + 16);
        }
        if ( v57 != v55 + 32 )
          _libc_free(v57);
      }
      while ( v54 != v55 );
      v54 = *((_QWORD *)v52 + 1);
    }
    if ( (char *)v54 != v52 + 24 )
      _libc_free(v54);
  }
  result = v75;
  *(_DWORD *)(v75 + 5568) = 0;
  return result;
}
