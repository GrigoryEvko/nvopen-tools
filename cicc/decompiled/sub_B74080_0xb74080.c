// Function: sub_B74080
// Address: 0xb74080
//
void __fastcall sub_B74080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // ebx
  unsigned int v7; // ebx
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // r15
  __int64 *v15; // r13
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // al
  char v20; // al
  unsigned int v21; // eax
  unsigned int v22; // r13d
  int v23; // r15d
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // ebx
  unsigned int v28; // eax
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rbx
  __int64 v36; // r8
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 m; // r12
  _QWORD *j; // rbx
  _QWORD *i; // r13
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rbx
  __int64 k; // r12
  _QWORD *n; // rbx
  __int64 v53; // [rsp+0h] [rbp-80h]
  __int64 v54; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v57; // [rsp+38h] [rbp-48h]

  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v6 )
  {
    a3 = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)a3 )
      return;
    v7 = *(_DWORD *)(a1 + 24);
    if ( v7 <= 0x40 )
    {
LABEL_4:
      v8 = sub_C33690(a1, a2, a3, a4, a5);
      v12 = sub_C33340(a1, a2, v9, v10, v11);
      v13 = v12;
      if ( v8 == v12 )
      {
        sub_C3C5A0(&v54, v12, 1);
        sub_C3C5A0(&v56, v13, 2);
      }
      else
      {
        sub_C36740(&v54, v8, 1);
        sub_C36740(&v56, v8, 2);
      }
      v14 = *(__int64 **)(a1 + 8);
      v15 = &v14[4 * *(unsigned int *)(a1 + 24)];
      if ( v15 == v14 )
      {
LABEL_28:
        if ( v13 == v56 )
        {
          if ( v57 )
          {
            for ( i = &v57[3 * *(v57 - 1)]; v57 != i; sub_91D830(i) )
              i -= 3;
            j_j_j___libc_free_0_0(i - 1);
          }
        }
        else
        {
          sub_C338F0(&v56);
        }
        *(_QWORD *)(a1 + 16) = 0;
        if ( v13 == v54 )
        {
          if ( v55 )
          {
            for ( j = &v55[3 * *(v55 - 1)]; v55 != j; sub_91D830(j) )
              j -= 3;
            j_j_j___libc_free_0_0(j - 1);
          }
        }
        else
        {
          sub_C338F0(&v54);
        }
        return;
      }
      while ( 1 )
      {
        v18 = *v14;
        if ( *v14 == v54 )
        {
          if ( v13 == v18 )
            v19 = sub_C3E590(v14);
          else
            v19 = sub_C33D00(v14);
          if ( v19 )
            goto LABEL_14;
          v18 = *v14;
          if ( v56 == *v14 )
            goto LABEL_20;
LABEL_9:
          v16 = v14[3];
          if ( v16 )
          {
            v53 = v14[3];
            sub_91D830((_QWORD *)(v16 + 24));
            sub_BD7260(v53);
            sub_BD2DD0(v53);
          }
          v17 = v54;
          if ( v13 != *v14 )
            goto LABEL_12;
LABEL_24:
          if ( v17 == v13 )
          {
            sub_C3C9E0(v14, &v54);
            goto LABEL_14;
          }
LABEL_25:
          if ( v14 == &v54 )
            goto LABEL_14;
          sub_91D830(v14);
          if ( v13 == v54 )
          {
            sub_C3C790(v14, &v54);
            goto LABEL_14;
          }
          sub_C33EB0(v14, &v54);
          v14 += 4;
          if ( v15 == v14 )
            goto LABEL_28;
        }
        else
        {
          if ( v56 != v18 )
            goto LABEL_9;
LABEL_20:
          if ( v13 == v18 )
            v20 = sub_C3E590(v14);
          else
            v20 = sub_C33D00(v14);
          if ( !v20 )
            goto LABEL_9;
          v17 = v54;
          if ( v13 == *v14 )
            goto LABEL_24;
LABEL_12:
          if ( v17 == v13 )
            goto LABEL_25;
          sub_C33E70(v14, &v54);
LABEL_14:
          v14 += 4;
          if ( v15 == v14 )
            goto LABEL_28;
        }
      }
    }
    sub_B73DB0(a1, a2, a3, a4, a5);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * v7, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_64;
  }
  v21 = 4 * v6;
  v22 = *(_DWORD *)(a1 + 24);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v21 = 64;
  if ( v22 <= v21 )
    goto LABEL_4;
  v23 = 64;
  sub_B73DB0(a1, a2, a3, a4, a5);
  v27 = v6 - 1;
  if ( v27 )
  {
    _BitScanReverse(&v28, v27);
    v28 ^= 0x1Fu;
    v25 = 33 - v28;
    v23 = 1 << (33 - v28);
    if ( v23 < 64 )
      v23 = 64;
  }
  v29 = sub_C33690(a1, a2, v24, v25, v26);
  v35 = sub_C33340(a1, a2, v30, v31, v32);
  if ( v23 == *(_DWORD *)(a1 + 24) )
  {
LABEL_64:
    *(_QWORD *)(a1 + 16) = 0;
    v44 = sub_C33690(a1, a2, v33, v34, v36);
    v48 = sub_C33340(a1, a2, v45, v46, v47);
    v49 = v48;
    if ( v44 == v48 )
      sub_C3C5A0(&v56, v48, 1);
    else
      sub_C36740(&v56, v44, 1);
    v50 = *(_QWORD *)(a1 + 8);
    for ( k = v50 + 32LL * *(unsigned int *)(a1 + 24); k != v50; v50 += 32 )
    {
      if ( v50 )
      {
        if ( v49 == v56 )
          sub_C3C790(v50, &v56);
        else
          sub_C33EB0(v50, &v56);
      }
    }
    sub_91D830(&v56);
    return;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * v22, 8);
  v37 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
       | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
       | (4 * v23 / 3u + 1)
       | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
  v38 = (v37
       | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
         | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
         | (4 * v23 / 3u + 1)
         | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
       | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
       | (4 * v23 / 3u + 1)
       | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 24) = v38;
  v39 = sub_C7D670(32 * v38, 8);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v39;
  if ( v29 == v35 )
    sub_C3C5A0(&v56, v35, 1);
  else
    sub_C36740(&v56, v29, 1);
  v40 = *(_QWORD *)(a1 + 8);
  for ( m = v40 + 32LL * *(unsigned int *)(a1 + 24); m != v40; v40 += 32 )
  {
    if ( v40 )
    {
      if ( v56 == v35 )
        sub_C3C790(v40, &v56);
      else
        sub_C33EB0(v40, &v56);
    }
  }
  if ( v56 == v35 )
  {
    if ( v57 )
    {
      for ( n = &v57[3 * *(v57 - 1)]; v57 != n; sub_91D830(n) )
        n -= 3;
      j_j_j___libc_free_0_0(n - 1);
    }
  }
  else
  {
    sub_C338F0(&v56);
  }
}
