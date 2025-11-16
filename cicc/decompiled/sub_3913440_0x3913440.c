// Function: sub_3913440
// Address: 0x3913440
//
__int64 __fastcall sub_3913440(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *j; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 result; // rax
  unsigned int v15; // ecx
  _QWORD *v16; // rdi
  unsigned int v17; // eax
  int v18; // eax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  int v21; // ebx
  unsigned __int64 v22; // r13
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v26; // rdi
  int v27; // edx
  int v28; // ebx
  unsigned int v29; // r14d
  unsigned int v30; // eax
  _QWORD *v31; // rdi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *i; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rax

  v2 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  if ( !v2 && !*(_DWORD *)(a1 + 36) )
    goto LABEL_15;
  v3 = *(_QWORD **)(a1 + 24);
  v4 = 4 * v2;
  v5 = *(unsigned int *)(a1 + 40);
  v6 = &v3[4 * v5];
  if ( (unsigned int)(4 * v2) < 0x40 )
    v4 = 64;
  if ( (unsigned int)v5 <= v4 )
  {
    while ( v3 != v6 )
    {
      if ( *v3 != -8 )
      {
        if ( *v3 != -16 )
        {
          v7 = v3[1];
          if ( v7 )
            j_j___libc_free_0(v7);
        }
        *v3 = -8;
      }
      v3 += 4;
    }
    goto LABEL_14;
  }
  do
  {
    while ( *v3 == -16 )
    {
LABEL_43:
      v3 += 4;
      if ( v3 == v6 )
        goto LABEL_47;
    }
    if ( *v3 != -8 )
    {
      v26 = v3[1];
      if ( v26 )
        j_j___libc_free_0(v26);
      goto LABEL_43;
    }
    v3 += 4;
  }
  while ( v3 != v6 );
LABEL_47:
  v27 = *(_DWORD *)(a1 + 40);
  if ( !v2 )
  {
    if ( v27 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 24));
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      goto LABEL_15;
    }
LABEL_14:
    *(_QWORD *)(a1 + 32) = 0;
    goto LABEL_15;
  }
  v28 = 64;
  v29 = v2 - 1;
  if ( v29 )
  {
    _BitScanReverse(&v30, v29);
    v28 = 1 << (33 - (v30 ^ 0x1F));
    if ( v28 < 64 )
      v28 = 64;
  }
  v31 = *(_QWORD **)(a1 + 24);
  if ( v28 == v27 )
  {
    *(_QWORD *)(a1 + 32) = 0;
    v38 = &v31[4 * (unsigned int)v28];
    do
    {
      if ( v31 )
        *v31 = -8;
      v31 += 4;
    }
    while ( v38 != v31 );
  }
  else
  {
    j___libc_free_0((unsigned __int64)v31);
    v32 = ((((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
             | (4 * v28 / 3u + 1)
             | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
           | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 16;
    v33 = (v32
         | (((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
             | (4 * v28 / 3u + 1)
             | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
           | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 40) = v33;
    v34 = (_QWORD *)sub_22077B0(32 * v33);
    v35 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 24) = v34;
    for ( i = &v34[4 * v35]; i != v34; v34 += 4 )
    {
      if ( v34 )
        *v34 = -8;
    }
  }
LABEL_15:
  v8 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( v8 )
  {
    v15 = 4 * v8;
    v9 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v15 = 64;
    if ( v15 >= (unsigned int)v9 )
    {
LABEL_18:
      v10 = *(_QWORD **)(a1 + 56);
      for ( j = &v10[2 * v9]; j != v10; v10 += 2 )
        *v10 = -8;
      *(_QWORD *)(a1 + 64) = 0;
      goto LABEL_21;
    }
    v16 = *(_QWORD **)(a1 + 56);
    v17 = v8 - 1;
    if ( v17 )
    {
      _BitScanReverse(&v17, v17);
      v18 = 1 << (33 - (v17 ^ 0x1F));
      if ( v18 < 64 )
        v18 = 64;
      if ( (_DWORD)v9 == v18 )
      {
        *(_QWORD *)(a1 + 64) = 0;
        v37 = &v16[2 * (unsigned int)v9];
        do
        {
          if ( v16 )
            *v16 = -8;
          v16 += 2;
        }
        while ( v37 != v16 );
        goto LABEL_21;
      }
      v19 = (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
          | (4 * v18 / 3u + 1)
          | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)
          | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
            | (4 * v18 / 3u + 1)
            | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4);
      v20 = (v19 >> 8) | v19;
      v21 = (v20 | (v20 >> 16)) + 1;
      v22 = 16 * ((v20 | (v20 >> 16)) + 1);
    }
    else
    {
      v22 = 2048;
      v21 = 128;
    }
    j___libc_free_0((unsigned __int64)v16);
    *(_DWORD *)(a1 + 72) = v21;
    v23 = (_QWORD *)sub_22077B0(v22);
    v24 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v23;
    for ( k = &v23[2 * v24]; k != v23; v23 += 2 )
    {
      if ( v23 )
        *v23 = -8;
    }
    goto LABEL_21;
  }
  if ( *(_DWORD *)(a1 + 68) )
  {
    v9 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v9 <= 0x40 )
      goto LABEL_18;
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    *(_DWORD *)(a1 + 72) = 0;
  }
LABEL_21:
  sub_167FC70(a1 + 112);
  v12 = *(_QWORD *)(a1 + 168);
  if ( v12 != *(_QWORD *)(a1 + 176) )
    *(_QWORD *)(a1 + 176) = v12;
  v13 = *(_QWORD *)(a1 + 192);
  if ( v13 != *(_QWORD *)(a1 + 200) )
    *(_QWORD *)(a1 + 200) = v13;
  result = *(_QWORD *)(a1 + 216);
  if ( result != *(_QWORD *)(a1 + 224) )
    *(_QWORD *)(a1 + 224) = result;
  return result;
}
