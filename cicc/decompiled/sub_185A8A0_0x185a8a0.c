// Function: sub_185A8A0
// Address: 0x185a8a0
//
_QWORD *__fastcall sub_185A8A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  int v6; // eax
  unsigned int v7; // ecx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rcx
  _QWORD **v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // r15
  _QWORD *v18; // r12
  __int64 v19; // rdi
  _QWORD *v20; // r12
  _QWORD *v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rax
  int v24; // edx
  int v25; // r12d
  unsigned int v26; // eax
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *k; // rdx
  _QWORD *v33; // rdi
  unsigned int v34; // eax
  int v35; // eax
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  int v38; // r14d
  __int64 v39; // r12
  _QWORD *v40; // rax
  __int64 v41; // rdx
  _QWORD *j; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  int v45; // [rsp+Ch] [rbp-44h]
  _QWORD **v47; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  v3 = (_QWORD *)sub_22077B0(16);
  v4 = v3;
  if ( v3 )
  {
    v3[1] = v2;
    *v3 = &unk_49F13A0;
    goto LABEL_3;
  }
  if ( !v2 )
    goto LABEL_3;
  v6 = *(_DWORD *)(v2 + 80);
  ++*(_QWORD *)(v2 + 64);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(v2 + 84) )
      goto LABEL_12;
    v8 = *(unsigned int *)(v2 + 88);
    if ( (unsigned int)v8 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v2 + 72));
      *(_QWORD *)(v2 + 72) = 0;
      *(_QWORD *)(v2 + 80) = 0;
      *(_DWORD *)(v2 + 88) = 0;
      goto LABEL_12;
    }
    goto LABEL_9;
  }
  v7 = 4 * v6;
  v8 = *(unsigned int *)(v2 + 88);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v8 <= v7 )
  {
LABEL_9:
    v9 = *(_QWORD **)(v2 + 72);
    for ( i = &v9[3 * v8]; i != v9; *(v9 - 2) = -8 )
    {
      *v9 = -8;
      v9 += 3;
    }
    *(_QWORD *)(v2 + 80) = 0;
    goto LABEL_12;
  }
  v33 = *(_QWORD **)(v2 + 72);
  v34 = v6 - 1;
  if ( !v34 )
  {
    v39 = 3072;
    v38 = 128;
LABEL_57:
    j___libc_free_0(v33);
    *(_DWORD *)(v2 + 88) = v38;
    v40 = (_QWORD *)sub_22077B0(v39);
    v41 = *(unsigned int *)(v2 + 88);
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 72) = v40;
    for ( j = &v40[3 * v41]; j != v40; v40 += 3 )
    {
      if ( v40 )
      {
        *v40 = -8;
        v40[1] = -8;
      }
    }
    goto LABEL_12;
  }
  _BitScanReverse(&v34, v34);
  v35 = 1 << (33 - (v34 ^ 0x1F));
  if ( v35 < 64 )
    v35 = 64;
  if ( (_DWORD)v8 != v35 )
  {
    v36 = ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)
        | (4 * v35 / 3u + 1)
        | ((((unsigned __int64)(4 * v35 / 3u + 1) >> 1) | (4 * v35 / 3u + 1)) >> 2);
    v37 = (((v36 | (v36 >> 4)) >> 8) | v36 | (v36 >> 4) | ((((v36 | (v36 >> 4)) >> 8) | v36 | (v36 >> 4)) >> 16)) + 1;
    v38 = v37;
    v39 = 24 * v37;
    goto LABEL_57;
  }
  *(_QWORD *)(v2 + 80) = 0;
  v43 = &v33[3 * v8];
  do
  {
    if ( v33 )
    {
      *v33 = -8;
      v33[1] = -8;
    }
    v33 += 3;
  }
  while ( v43 != v33 );
LABEL_12:
  v11 = *(_DWORD *)(v2 + 48);
  ++*(_QWORD *)(v2 + 32);
  v45 = v11;
  if ( v11 || *(_DWORD *)(v2 + 52) )
  {
    v12 = *(_QWORD *)(v2 + 40);
    v13 = 4 * v11;
    v14 = *(unsigned int *)(v2 + 56);
    v15 = (_QWORD **)(v12 + 8);
    if ( (unsigned int)(4 * v45) < 0x40 )
      v13 = 64;
    v47 = (_QWORD **)(v12 + 32 * v14);
    if ( (unsigned int)v14 <= v13 )
    {
      if ( v12 + 32 * v14 != v12 )
      {
        while ( 1 )
        {
          v16 = (__int64)*(v15 - 1);
          if ( v16 != -8 )
          {
            if ( v16 != -16 )
            {
              v17 = *v15;
              while ( v17 != v15 )
              {
                v18 = v17;
                v17 = (_QWORD *)*v17;
                v19 = v18[3];
                if ( v19 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
                j_j___libc_free_0(v18, 32);
              }
            }
            *(v15 - 1) = (_QWORD *)-8LL;
          }
          if ( v47 == v15 + 3 )
            break;
          v15 += 4;
        }
      }
LABEL_39:
      *(_QWORD *)(v2 + 48) = 0;
      goto LABEL_3;
    }
    while ( 1 )
    {
      v23 = (__int64)*(v15 - 1);
      if ( v23 != -16 && v23 != -8 )
      {
        v20 = *v15;
        while ( v20 != v15 )
        {
          v21 = v20;
          v20 = (_QWORD *)*v20;
          v22 = v21[3];
          if ( v22 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v22 + 8LL))(v22);
          j_j___libc_free_0(v21, 32);
        }
      }
      if ( v47 == v15 + 3 )
        break;
      v15 += 4;
    }
    v24 = *(_DWORD *)(v2 + 56);
    if ( !v45 )
    {
      if ( v24 )
      {
        j___libc_free_0(*(_QWORD *)(v2 + 40));
        *(_QWORD *)(v2 + 40) = 0;
        *(_QWORD *)(v2 + 48) = 0;
        *(_DWORD *)(v2 + 56) = 0;
        goto LABEL_3;
      }
      goto LABEL_39;
    }
    v25 = 64;
    if ( v45 != 1 )
    {
      _BitScanReverse(&v26, v45 - 1);
      v25 = 1 << (33 - (v26 ^ 0x1F));
      if ( v25 < 64 )
        v25 = 64;
    }
    v27 = *(_QWORD **)(v2 + 40);
    if ( v25 == v24 )
    {
      *(_QWORD *)(v2 + 48) = 0;
      v44 = &v27[4 * (unsigned int)v25];
      do
      {
        if ( v27 )
          *v27 = -8;
        v27 += 4;
      }
      while ( v44 != v27 );
    }
    else
    {
      j___libc_free_0(v27);
      v28 = ((((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
               | (4 * v25 / 3u + 1)
               | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
             | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 16;
      v29 = (v28
           | (((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
               | (4 * v25 / 3u + 1)
               | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
             | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v2 + 56) = v29;
      v30 = (_QWORD *)sub_22077B0(32 * v29);
      v31 = *(unsigned int *)(v2 + 56);
      *(_QWORD *)(v2 + 48) = 0;
      *(_QWORD *)(v2 + 40) = v30;
      for ( k = &v30[4 * v31]; k != v30; v30 += 4 )
      {
        if ( v30 )
          *v30 = -8;
      }
    }
  }
LABEL_3:
  *a1 = v4;
  return a1;
}
