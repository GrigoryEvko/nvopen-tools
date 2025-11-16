// Function: sub_124C180
// Address: 0x124c180
//
__int64 __fastcall sub_124C180(__int64 a1)
{
  bool v2; // zf
  int v3; // r15d
  unsigned int v4; // eax
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r14
  _QWORD *v8; // r13
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *j; // rdx
  unsigned int v15; // ecx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  int v18; // ebx
  _QWORD *v19; // rax
  __int64 v20; // rdi
  unsigned int v21; // edx
  int v22; // ebx
  unsigned int v23; // r15d
  unsigned int v24; // eax
  _QWORD *v25; // rdi
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdi
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *i; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *k; // rdx
  _QWORD *v36; // rax

  v2 = *(_BYTE *)(a1 + 203) == 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 201) = 0;
  if ( !v2 )
    *(_BYTE *)(a1 + 203) = 0;
  v3 = *(_DWORD *)(a1 + 152);
  ++*(_QWORD *)(a1 + 136);
  if ( v3 || *(_DWORD *)(a1 + 156) )
  {
    v4 = 4 * v3;
    v5 = *(_QWORD **)(a1 + 144);
    v6 = *(unsigned int *)(a1 + 160);
    v7 = 32 * v6;
    if ( (unsigned int)(4 * v3) < 0x40 )
      v4 = 64;
    v8 = &v5[(unsigned __int64)v7 / 8];
    if ( (unsigned int)v6 <= v4 )
    {
      for ( ; v5 != v8; v5 += 4 )
      {
        if ( *v5 != -4096 )
        {
          if ( *v5 != -8192 )
          {
            v9 = v5[1];
            if ( v9 )
              j_j___libc_free_0(v9, v5[3] - v9);
          }
          *v5 = -4096;
        }
      }
      goto LABEL_15;
    }
    while ( 1 )
    {
      while ( *v5 == -8192 )
      {
LABEL_37:
        v5 += 4;
        if ( v8 == v5 )
          goto LABEL_41;
      }
      if ( *v5 != -4096 )
      {
        v20 = v5[1];
        if ( v20 )
          j_j___libc_free_0(v20, v5[3] - v20);
        goto LABEL_37;
      }
      v5 += 4;
      if ( v8 == v5 )
      {
LABEL_41:
        v21 = *(_DWORD *)(a1 + 160);
        if ( v3 )
        {
          v22 = 64;
          v23 = v3 - 1;
          if ( v23 )
          {
            _BitScanReverse(&v24, v23);
            v22 = 1 << (33 - (v24 ^ 0x1F));
            if ( v22 < 64 )
              v22 = 64;
          }
          v25 = *(_QWORD **)(a1 + 144);
          if ( v21 == v22 )
          {
            *(_QWORD *)(a1 + 152) = 0;
            v36 = &v25[4 * v21];
            do
            {
              if ( v25 )
                *v25 = -4096;
              v25 += 4;
            }
            while ( v36 != v25 );
          }
          else
          {
            sub_C7D6A0((__int64)v25, v7, 8);
            v26 = ((((((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                     | (4 * v22 / 3u + 1)
                     | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                   | (4 * v22 / 3u + 1)
                   | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                   | (4 * v22 / 3u + 1)
                   | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                 | (4 * v22 / 3u + 1)
                 | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 16;
            v27 = (v26
                 | (((((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                     | (4 * v22 / 3u + 1)
                     | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                   | (4 * v22 / 3u + 1)
                   | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                   | (4 * v22 / 3u + 1)
                   | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
                 | (4 * v22 / 3u + 1)
                 | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 160) = v27;
            v28 = (_QWORD *)sub_C7D670(32 * v27, 8);
            v29 = *(unsigned int *)(a1 + 160);
            *(_QWORD *)(a1 + 152) = 0;
            *(_QWORD *)(a1 + 144) = v28;
            for ( i = &v28[4 * v29]; i != v28; v28 += 4 )
            {
              if ( v28 )
                *v28 = -4096;
            }
          }
          break;
        }
        if ( v21 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 144), v7, 8);
          *(_QWORD *)(a1 + 144) = 0;
          *(_QWORD *)(a1 + 152) = 0;
          *(_DWORD *)(a1 + 160) = 0;
          break;
        }
LABEL_15:
        *(_QWORD *)(a1 + 152) = 0;
        break;
      }
    }
  }
  v10 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 188) )
      goto LABEL_22;
    v11 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)v11 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 176), 16LL * (unsigned int)v11, 8);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 184) = 0;
      *(_DWORD *)(a1 + 192) = 0;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v15 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 192);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v15 = 64;
  if ( v15 >= (unsigned int)v11 )
  {
LABEL_19:
    v12 = *(_QWORD **)(a1 + 176);
    for ( j = &v12[2 * v11]; j != v12; v12 += 2 )
      *v12 = -4096;
    *(_QWORD *)(a1 + 184) = 0;
    goto LABEL_22;
  }
  v16 = v10 - 1;
  if ( v16 )
  {
    _BitScanReverse(&v16, v16);
    v17 = *(_QWORD **)(a1 + 176);
    v18 = 1 << (33 - (v16 ^ 0x1F));
    if ( v18 < 64 )
      v18 = 64;
    if ( v18 == (_DWORD)v11 )
    {
      *(_QWORD *)(a1 + 184) = 0;
      v19 = &v17[2 * (unsigned int)v18];
      do
      {
        if ( v17 )
          *v17 = -4096;
        v17 += 2;
      }
      while ( v19 != v17 );
      goto LABEL_22;
    }
  }
  else
  {
    v17 = *(_QWORD **)(a1 + 176);
    v18 = 64;
  }
  sub_C7D6A0((__int64)v17, 16LL * (unsigned int)v11, 8);
  v31 = ((((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
       | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
       | (4 * v18 / 3u + 1)
       | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 16;
  v32 = (v31
       | (((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
         | (4 * v18 / 3u + 1)
         | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
       | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
       | (4 * v18 / 3u + 1)
       | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 192) = v32;
  v33 = (_QWORD *)sub_C7D670(16 * v32, 8);
  v34 = *(unsigned int *)(a1 + 192);
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 176) = v33;
  for ( k = &v33[2 * v34]; k != v33; v33 += 2 )
  {
    if ( v33 )
      *v33 = -4096;
  }
LABEL_22:
  *(_DWORD *)(a1 + 216) = 0;
  return sub_E8EB90(a1);
}
