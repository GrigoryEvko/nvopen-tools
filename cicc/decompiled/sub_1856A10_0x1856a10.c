// Function: sub_1856A10
// Address: 0x1856a10
//
__int64 __fastcall sub_1856A10(_QWORD *a1)
{
  __int64 v2; // r13
  int v3; // eax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  int v8; // r12d
  __int64 v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r14
  _QWORD **v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r14
  __int64 v17; // rdi
  _QWORD **k; // rbx
  __int64 v20; // rax
  _QWORD *v21; // r14
  _QWORD *v22; // r8
  __int64 v23; // rdi
  int v24; // edx
  int v25; // ebx
  unsigned int v26; // r12d
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *m; // rdx
  _QWORD *v34; // rdi
  unsigned int v35; // eax
  int v36; // eax
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  int v39; // ebx
  __int64 v40; // r12
  _QWORD *v41; // rax
  __int64 v42; // rdx
  _QWORD *j; // rdx
  _QWORD *v44; // rax
  _QWORD *v45; // rax
  _QWORD *v46; // [rsp+0h] [rbp-40h]
  _QWORD **v47; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  *a1 = &unk_49F13A0;
  if ( !v2 )
    return j_j___libc_free_0(a1, 16);
  v3 = *(_DWORD *)(v2 + 80);
  ++*(_QWORD *)(v2 + 64);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(v2 + 84) )
      goto LABEL_9;
    v5 = *(unsigned int *)(v2 + 88);
    if ( (unsigned int)v5 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v2 + 72));
      *(_QWORD *)(v2 + 72) = 0;
      *(_QWORD *)(v2 + 80) = 0;
      *(_DWORD *)(v2 + 88) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v4 = 4 * v3;
  v5 = *(unsigned int *)(v2 + 88);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v4 = 64;
  if ( (unsigned int)v5 <= v4 )
  {
LABEL_6:
    v6 = *(_QWORD **)(v2 + 72);
    for ( i = &v6[3 * v5]; i != v6; *(v6 - 2) = -8 )
    {
      *v6 = -8;
      v6 += 3;
    }
    *(_QWORD *)(v2 + 80) = 0;
    goto LABEL_9;
  }
  v34 = *(_QWORD **)(v2 + 72);
  v35 = v3 - 1;
  if ( !v35 )
  {
    v40 = 3072;
    v39 = 128;
LABEL_55:
    j___libc_free_0(v34);
    *(_DWORD *)(v2 + 88) = v39;
    v41 = (_QWORD *)sub_22077B0(v40);
    v42 = *(unsigned int *)(v2 + 88);
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 72) = v41;
    for ( j = &v41[3 * v42]; j != v41; v41 += 3 )
    {
      if ( v41 )
      {
        *v41 = -8;
        v41[1] = -8;
      }
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v35, v35);
  v36 = 1 << (33 - (v35 ^ 0x1F));
  if ( v36 < 64 )
    v36 = 64;
  if ( (_DWORD)v5 != v36 )
  {
    v37 = ((unsigned __int64)(4 * v36 / 3u + 1) >> 1)
        | (4 * v36 / 3u + 1)
        | ((((unsigned __int64)(4 * v36 / 3u + 1) >> 1) | (4 * v36 / 3u + 1)) >> 2);
    v38 = (((v37 | (v37 >> 4)) >> 8) | v37 | (v37 >> 4) | ((((v37 | (v37 >> 4)) >> 8) | v37 | (v37 >> 4)) >> 16)) + 1;
    v39 = v38;
    v40 = 24 * v38;
    goto LABEL_55;
  }
  *(_QWORD *)(v2 + 80) = 0;
  v44 = &v34[3 * v5];
  do
  {
    if ( v34 )
    {
      *v34 = -8;
      v34[1] = -8;
    }
    v34 += 3;
  }
  while ( v44 != v34 );
LABEL_9:
  v8 = *(_DWORD *)(v2 + 48);
  ++*(_QWORD *)(v2 + 32);
  if ( v8 || *(_DWORD *)(v2 + 52) )
  {
    v9 = *(_QWORD *)(v2 + 40);
    v10 = 4 * v8;
    v11 = *(unsigned int *)(v2 + 56);
    v12 = 32 * v11;
    if ( (unsigned int)(4 * v8) < 0x40 )
      v10 = 64;
    v47 = (_QWORD **)(v9 + v12);
    if ( (unsigned int)v11 <= v10 )
    {
      v13 = (_QWORD **)(v9 + 8);
      if ( v9 != v9 + v12 )
      {
        while ( 1 )
        {
          v14 = (__int64)*(v13 - 1);
          if ( v14 != -8 )
          {
            if ( v14 != -16 )
            {
              v15 = *v13;
              while ( v15 != v13 )
              {
                v16 = v15;
                v15 = (_QWORD *)*v15;
                v17 = v16[3];
                if ( v17 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
                j_j___libc_free_0(v16, 32);
              }
            }
            *(v13 - 1) = (_QWORD *)-8LL;
          }
          if ( v47 == v13 + 3 )
            break;
          v13 += 4;
        }
      }
LABEL_24:
      *(_QWORD *)(v2 + 48) = 0;
      return j_j___libc_free_0(a1, 16);
    }
    for ( k = (_QWORD **)(v9 + 8); ; k += 4 )
    {
      v20 = (__int64)*(k - 1);
      if ( v20 != -8 && v20 != -16 )
      {
        v21 = *k;
        while ( k != v21 )
        {
          v22 = v21;
          v21 = (_QWORD *)*v21;
          v23 = v22[3];
          if ( v23 )
          {
            v46 = v22;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
            v22 = v46;
          }
          j_j___libc_free_0(v22, 32);
        }
      }
      if ( v47 == k + 3 )
        break;
    }
    v24 = *(_DWORD *)(v2 + 56);
    if ( !v8 )
    {
      if ( v24 )
      {
        j___libc_free_0(*(_QWORD *)(v2 + 40));
        *(_QWORD *)(v2 + 40) = 0;
        *(_QWORD *)(v2 + 48) = 0;
        *(_DWORD *)(v2 + 56) = 0;
        return j_j___libc_free_0(a1, 16);
      }
      goto LABEL_24;
    }
    v25 = 64;
    v26 = v8 - 1;
    if ( v26 )
    {
      _BitScanReverse(&v27, v26);
      v25 = 1 << (33 - (v27 ^ 0x1F));
      if ( v25 < 64 )
        v25 = 64;
    }
    v28 = *(_QWORD **)(v2 + 40);
    if ( v25 == v24 )
    {
      *(_QWORD *)(v2 + 48) = 0;
      v45 = &v28[4 * (unsigned int)v25];
      do
      {
        if ( v28 )
          *v28 = -8;
        v28 += 4;
      }
      while ( v45 != v28 );
    }
    else
    {
      j___libc_free_0(v28);
      v29 = ((((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
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
      v30 = (v29
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
      *(_DWORD *)(v2 + 56) = v30;
      v31 = (_QWORD *)sub_22077B0(32 * v30);
      v32 = *(unsigned int *)(v2 + 56);
      *(_QWORD *)(v2 + 48) = 0;
      *(_QWORD *)(v2 + 40) = v31;
      for ( m = &v31[4 * v32]; m != v31; v31 += 4 )
      {
        if ( v31 )
          *v31 = -8;
      }
    }
  }
  return j_j___libc_free_0(a1, 16);
}
