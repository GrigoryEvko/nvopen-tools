// Function: sub_DDF4E0
// Address: 0xddf4e0
//
__int64 *__fastcall sub_DDF4E0(__int64 a1, __int64 **a2, char *a3)
{
  __int64 v3; // r14
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 ***v9; // r15
  __int64 v10; // r9
  int v11; // r10d
  unsigned int v12; // edi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  char **v16; // rax
  char ***v17; // r15
  int v18; // esi
  char **v19; // rdx
  __int64 *result; // rax
  int v21; // edi
  int v22; // ecx
  unsigned int v23; // esi
  int v24; // r11d
  __int64 v25; // r9
  unsigned int v26; // r8d
  __int64 v27; // rcx
  __int64 ***v28; // rdx
  __int64 **v29; // rdi
  __int64 v30; // rsi
  __int64 i; // rdx
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // r12
  __int64 v35; // rax
  char **v36; // rax
  int v37; // eax
  int v38; // esi
  __int64 v39; // rdi
  unsigned int v40; // eax
  __int64 **v41; // rdx
  int v42; // r8d
  __int64 ***v43; // r9
  int v44; // edx
  int v45; // esi
  int v46; // r8d
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 **v49; // rdx
  int v50; // eax
  int v51; // ecx
  int v52; // eax
  int v53; // r10d
  __int64 v54; // r9
  unsigned int v55; // eax
  __int64 **v56; // r8
  int v57; // edi
  __int64 ***v58; // rsi
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rdi
  int v61; // eax
  int v62; // r8d
  __int64 v63; // r9
  int v64; // esi
  __int64 v65; // r15
  __int64 **v66; // rdi
  __int64 ***v67; // rax
  __int64 *v68; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 776;
  v7 = *(_DWORD *)(a1 + 800);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 776);
    goto LABEL_37;
  }
  v8 = *(_QWORD *)(a1 + 784);
  v9 = 0;
  v10 = v7 - 1;
  v11 = 1;
  v12 = v10 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v13 = v8 + 56LL * v12;
  v14 = *(_QWORD *)v13;
  if ( *(__int64 ***)v13 != a2 )
  {
    while ( v14 != -4096 )
    {
      if ( v14 == -8192 && !v9 )
        v9 = (__int64 ***)v13;
      v12 = v10 & (v11 + v12);
      v13 = v8 + 56LL * v12;
      v14 = *(_QWORD *)v13;
      if ( *(__int64 ***)v13 == a2 )
        goto LABEL_3;
      ++v11;
    }
    v21 = *(_DWORD *)(a1 + 792);
    if ( !v9 )
      v9 = (__int64 ***)v13;
    ++*(_QWORD *)(a1 + 776);
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 796) - v22 > v7 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(a1 + 792) = v22;
        if ( *v9 != (__int64 **)-4096LL )
          --*(_DWORD *)(a1 + 796);
        v19 = (char **)(v9 + 3);
        *v9 = a2;
        v17 = (char ***)(v9 + 1);
        *v17 = v19;
        v17[1] = (char **)0x200000000LL;
        goto LABEL_23;
      }
      sub_DA9EE0(v3, v7);
      v44 = *(_DWORD *)(a1 + 800);
      if ( v44 )
      {
        v45 = v44 - 1;
        v46 = 1;
        v43 = 0;
        v47 = *(_QWORD *)(a1 + 784);
        LODWORD(v48) = (v44 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
        v9 = (__int64 ***)(v47 + 56LL * (unsigned int)v48);
        v49 = *v9;
        v22 = *(_DWORD *)(a1 + 792) + 1;
        if ( *v9 == a2 )
          goto LABEL_20;
        while ( v49 != (__int64 **)-4096LL )
        {
          if ( v49 == (__int64 **)-8192LL && !v43 )
            v43 = v9;
          v48 = v45 & (unsigned int)(v48 + v46);
          v9 = (__int64 ***)(v47 + 56 * v48);
          v49 = *v9;
          if ( *v9 == a2 )
            goto LABEL_20;
          ++v46;
        }
        goto LABEL_41;
      }
      goto LABEL_101;
    }
LABEL_37:
    sub_DA9EE0(v3, 2 * v7);
    v37 = *(_DWORD *)(a1 + 800);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 784);
      v40 = (v37 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 ***)(v39 + 56LL * v40);
      v41 = *v9;
      v22 = *(_DWORD *)(a1 + 792) + 1;
      if ( *v9 == a2 )
        goto LABEL_20;
      v42 = 1;
      v43 = 0;
      while ( v41 != (__int64 **)-4096LL )
      {
        if ( !v43 && v41 == (__int64 **)-8192LL )
          v43 = v9;
        v40 = v38 & (v42 + v40);
        v9 = (__int64 ***)(v39 + 56LL * v40);
        v41 = *v9;
        if ( *v9 == a2 )
          goto LABEL_20;
        ++v42;
      }
LABEL_41:
      if ( v43 )
        v9 = v43;
      goto LABEL_20;
    }
LABEL_101:
    ++*(_DWORD *)(a1 + 792);
    BUG();
  }
LABEL_3:
  v15 = *(unsigned int *)(v13 + 16);
  v16 = *(char ***)(v13 + 8);
  v17 = (char ***)(v13 + 8);
  v18 = *(_DWORD *)(v13 + 16);
  v19 = &v16[2 * v15];
  if ( v19 == v16 )
  {
LABEL_73:
    v59 = *(unsigned int *)(v13 + 20);
    if ( v15 >= v59 )
    {
      v60 = v15 + 1;
      if ( v60 > v59 )
      {
        sub_C8D5F0((__int64)v17, (const void *)(v13 + 24), v60, 0x10u, v8, v10);
        v19 = &(*v17)[2 * *((unsigned int *)v17 + 2)];
      }
      *v19 = a3;
      v19[1] = 0;
      ++*((_DWORD *)v17 + 2);
LABEL_25:
      result = sub_DDEB00((__int64 *)a1, a2, a3);
      v23 = *(_DWORD *)(a1 + 800);
      v68 = result;
      if ( v23 )
      {
        v24 = 1;
        v25 = *(_QWORD *)(a1 + 784);
        v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v27 = v25 + 56LL * v26;
        v28 = 0;
        v29 = *(__int64 ***)v27;
        if ( *(__int64 ***)v27 == a2 )
        {
LABEL_27:
          v30 = *(_QWORD *)(v27 + 8);
          for ( i = v30 + 16LL * *(unsigned int *)(v27 + 16); i != v30; i -= 16 )
          {
            if ( *(char **)(i - 16) == a3 )
            {
              *(_QWORD *)(i - 8) = result;
              if ( *((_WORD *)result + 12) )
              {
                v34 = sub_DAA240(a1 + 808, &v68);
                v35 = *((unsigned int *)v34 + 2);
                if ( v35 + 1 > (unsigned __int64)*((unsigned int *)v34 + 3) )
                {
                  sub_C8D5F0((__int64)v34, v34 + 2, v35 + 1, 0x10u, v32, v33);
                  v35 = *((unsigned int *)v34 + 2);
                }
                v36 = (char **)(*v34 + 16 * v35);
                *v36 = a3;
                v36[1] = (char *)a2;
                ++*((_DWORD *)v34 + 2);
              }
              return v68;
            }
          }
          return result;
        }
        while ( v29 != (__int64 **)-4096LL )
        {
          if ( !v28 && v29 == (__int64 **)-8192LL )
            v28 = (__int64 ***)v27;
          v26 = (v23 - 1) & (v24 + v26);
          v27 = v25 + 56LL * v26;
          v29 = *(__int64 ***)v27;
          if ( *(__int64 ***)v27 == a2 )
            goto LABEL_27;
          ++v24;
        }
        v50 = *(_DWORD *)(a1 + 792);
        if ( !v28 )
          v28 = (__int64 ***)v27;
        ++*(_QWORD *)(a1 + 776);
        v51 = v50 + 1;
        if ( 4 * (v50 + 1) < 3 * v23 )
        {
          if ( v23 - *(_DWORD *)(a1 + 796) - v51 > v23 >> 3 )
          {
LABEL_62:
            *(_DWORD *)(a1 + 792) = v51;
            if ( *v28 != (__int64 **)-4096LL )
              --*(_DWORD *)(a1 + 796);
            *v28 = a2;
            v28[1] = (__int64 **)(v28 + 3);
            v28[2] = (__int64 **)0x200000000LL;
            return v68;
          }
          sub_DA9EE0(v3, v23);
          v61 = *(_DWORD *)(a1 + 800);
          if ( v61 )
          {
            v62 = v61 - 1;
            v63 = *(_QWORD *)(a1 + 784);
            v64 = 1;
            LODWORD(v65) = (v61 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v28 = (__int64 ***)(v63 + 56LL * (unsigned int)v65);
            v66 = *v28;
            v51 = *(_DWORD *)(a1 + 792) + 1;
            v67 = 0;
            if ( *v28 != a2 )
            {
              while ( v66 != (__int64 **)-4096LL )
              {
                if ( !v67 && v66 == (__int64 **)-8192LL )
                  v67 = v28;
                v65 = v62 & (unsigned int)(v65 + v64);
                v28 = (__int64 ***)(v63 + 56 * v65);
                v66 = *v28;
                if ( *v28 == a2 )
                  goto LABEL_62;
                ++v64;
              }
              if ( v67 )
                v28 = v67;
            }
            goto LABEL_62;
          }
LABEL_100:
          ++*(_DWORD *)(a1 + 792);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 776);
      }
      sub_DA9EE0(v3, 2 * v23);
      v52 = *(_DWORD *)(a1 + 800);
      if ( v52 )
      {
        v53 = v52 - 1;
        v54 = *(_QWORD *)(a1 + 784);
        v55 = (v52 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v51 = *(_DWORD *)(a1 + 792) + 1;
        v28 = (__int64 ***)(v54 + 56LL * v55);
        v56 = *v28;
        if ( *v28 != a2 )
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != (__int64 **)-4096LL )
          {
            if ( v56 == (__int64 **)-8192LL && !v58 )
              v58 = v28;
            v55 = v53 & (v57 + v55);
            v28 = (__int64 ***)(v54 + 56LL * v55);
            v56 = *v28;
            if ( *v28 == a2 )
              goto LABEL_62;
            ++v57;
          }
          if ( v58 )
            v28 = v58;
        }
        goto LABEL_62;
      }
      goto LABEL_100;
    }
    if ( !v19 )
    {
LABEL_24:
      *((_DWORD *)v17 + 2) = v18 + 1;
      goto LABEL_25;
    }
LABEL_23:
    *v19 = a3;
    v19[1] = 0;
    v18 = *((_DWORD *)v17 + 2);
    goto LABEL_24;
  }
  while ( *v16 != a3 )
  {
    v16 += 2;
    if ( v19 == v16 )
      goto LABEL_73;
  }
  result = (__int64 *)v16[1];
  if ( !result )
    return (__int64 *)a2;
  return result;
}
