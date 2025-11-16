// Function: sub_FD6240
// Address: 0xfd6240
//
__int64 __fastcall sub_FD6240(__int64 a1, unsigned __int64 a2)
{
  int v3; // r15d
  unsigned int v4; // eax
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r12
  _QWORD *v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 *v11; // rbx
  __int64 result; // rax
  unsigned __int64 *v13; // r15
  unsigned __int64 v14; // rdx
  _QWORD *v15; // r14
  _QWORD *v16; // r12
  __int64 v17; // rax
  unsigned __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // eax
  unsigned __int64 *v22; // r14
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rdx
  __int64 v28; // rdi
  unsigned __int64 *v29; // rbx
  __int64 v30; // rax
  bool v31; // zf
  _QWORD v32[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+30h] [rbp-50h] BYREF
  __int64 v35; // [rsp+38h] [rbp-48h]
  __int64 i; // [rsp+40h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( !v3 && !*(_DWORD *)(a1 + 44) )
    goto LABEL_19;
  v4 = 4 * v3;
  v5 = *(_QWORD **)(a1 + 32);
  v6 = *(unsigned int *)(a1 + 48);
  v7 = 32 * v6;
  if ( (unsigned int)(4 * v3) < 0x40 )
    v4 = 64;
  v8 = &v5[(unsigned __int64)v7 / 8];
  if ( (unsigned int)v6 > v4 )
  {
    v32[0] = 0;
    v32[1] = 0;
    v33 = -4096;
    v34 = 0;
    v35 = 0;
    i = -8192;
    do
    {
      v19 = v5[2];
      if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        sub_BD60C0(v5);
      v5 += 4;
    }
    while ( v5 != v8 );
    if ( v33 != 0 && v33 != -4096 )
      sub_BD60C0(v32);
    if ( v3 )
    {
      v20 = 64;
      if ( v3 != 1 )
      {
        _BitScanReverse(&v21, v3 - 1);
        v20 = (unsigned int)(1 << (33 - (v21 ^ 0x1F)));
        if ( (int)v20 < 64 )
          v20 = 64;
      }
      v22 = *(unsigned __int64 **)(a1 + 32);
      if ( *(_DWORD *)(a1 + 48) == (_DWORD)v20 )
      {
        *(_QWORD *)(a1 + 40) = 0;
        v29 = &v22[4 * v20];
        v34 = 0;
        v35 = 0;
        i = -4096;
        if ( v29 != v22 )
        {
          do
          {
            if ( v22 )
            {
              *v22 = 0;
              v22[1] = 0;
              v30 = i;
              v31 = i == 0;
              v22[2] = i;
              if ( v30 != -4096 && !v31 && v30 != -8192 )
              {
                a2 = v34 & 0xFFFFFFFFFFFFFFF8LL;
                sub_BD6050(v22, v34 & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
            v22 += 4;
          }
          while ( v29 != v22 );
          if ( i != 0 && i != -4096 && i != -8192 )
            goto LABEL_18;
        }
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 32), v7, 8);
        a2 = 8;
        v23 = ((((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v20 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v20 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 16;
        v24 = (v23
             | (((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v20 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v20 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 48) = v24;
        v25 = (_QWORD *)sub_C7D670(32 * v24, 8);
        v26 = *(unsigned int *)(a1 + 48);
        *(_QWORD *)(a1 + 40) = 0;
        *(_QWORD *)(a1 + 32) = v25;
        v34 = 0;
        v27 = &v25[4 * v26];
        v35 = 0;
        for ( i = -4096; v27 != v25; v25 += 4 )
        {
          if ( v25 )
          {
            *v25 = 0;
            v25[1] = 0;
            v25[2] = -4096;
          }
        }
      }
      goto LABEL_19;
    }
    v28 = *(_QWORD *)(a1 + 32);
    if ( *(_DWORD *)(a1 + 48) )
    {
      a2 = v7;
      sub_C7D6A0(v28, v7, 8);
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_DWORD *)(a1 + 48) = 0;
      goto LABEL_19;
    }
LABEL_32:
    *(_QWORD *)(a1 + 40) = 0;
    goto LABEL_19;
  }
  v34 = 0;
  v9 = -4096;
  v35 = 0;
  i = -4096;
  if ( v5 == v8 )
    goto LABEL_32;
  do
  {
    v10 = v5[2];
    if ( v10 != v9 )
    {
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
        sub_BD60C0(v5);
      v5[2] = v9;
      if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
        sub_BD73F0((__int64)v5);
      v9 = i;
    }
    v5 += 4;
  }
  while ( v5 != v8 );
  *(_QWORD *)(a1 + 40) = 0;
  if ( v9 != -4096 && v9 != 0 && v9 != -8192 )
LABEL_18:
    sub_BD60C0(&v34);
LABEL_19:
  v11 = *(unsigned __int64 **)(a1 + 16);
  for ( result = a1 + 8; (unsigned __int64 *)(a1 + 8) != v11; result = j_j___libc_free_0(v13, 72) )
  {
    v13 = v11;
    v11 = (unsigned __int64 *)v11[1];
    v14 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
    *v11 = v14 | *v11 & 7;
    *(_QWORD *)(v14 + 8) = v11;
    v15 = (_QWORD *)v13[6];
    v16 = (_QWORD *)v13[5];
    *v13 &= 7u;
    v13[1] = 0;
    if ( v15 != v16 )
    {
      do
      {
        v17 = v16[2];
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
          sub_BD60C0(v16);
        v16 += 3;
      }
      while ( v15 != v16 );
      v16 = (_QWORD *)v13[5];
    }
    if ( v16 )
    {
      a2 = v13[7] - (_QWORD)v16;
      j_j___libc_free_0(v16, a2);
    }
    v18 = (unsigned __int64 *)v13[3];
    if ( v13 + 5 != v18 )
      _libc_free(v18, a2);
    a2 = 72;
  }
  return result;
}
