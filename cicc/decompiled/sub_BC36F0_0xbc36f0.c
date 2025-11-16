// Function: sub_BC36F0
// Address: 0xbc36f0
//
void __fastcall sub_BC36F0(__int64 *a1)
{
  unsigned int v2; // r15d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r13
  _QWORD *v7; // r14
  __int64 v8; // r13
  __int64 v9; // rsi
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // rbx
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  int v20; // edx
  int v21; // ebx
  unsigned int v22; // r15d
  unsigned int v23; // eax
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *i; // rdx
  _QWORD *v30; // rax
  __int64 v31; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return;
  v2 = *((_DWORD *)a1 + 10);
  ++a1[3];
  if ( __PAIR64__(*((_DWORD *)a1 + 11), v2) )
  {
    v3 = (_QWORD *)a1[4];
    v4 = 4 * v2;
    v5 = *((unsigned int *)a1 + 12);
    v6 = 16 * v5;
    if ( 4 * v2 < 0x40 )
      v4 = 64;
    v7 = &v3[(unsigned __int64)v6 / 8];
    if ( (unsigned int)v5 <= v4 )
    {
      while ( v3 != v7 )
      {
        if ( *v3 != -4096 )
        {
          if ( *v3 != -8192 )
          {
            v8 = v3[1];
            if ( v8 )
            {
              sub_C9F8C0(v3[1]);
              j_j___libc_free_0(v8, 176);
            }
          }
          *v3 = -4096;
        }
        v3 += 2;
      }
    }
    else
    {
      do
      {
        if ( *v3 != -4096 && *v3 != -8192 )
        {
          v19 = v3[1];
          if ( v19 )
          {
            v31 = v3[1];
            sub_C9F8C0(v19);
            j_j___libc_free_0(v31, 176);
          }
        }
        v3 += 2;
      }
      while ( v3 != v7 );
      v20 = *((_DWORD *)a1 + 12);
      if ( v2 )
      {
        v21 = 64;
        v22 = v2 - 1;
        if ( v22 )
        {
          _BitScanReverse(&v23, v22);
          v21 = 1 << (33 - (v23 ^ 0x1F));
          if ( v21 < 64 )
            v21 = 64;
        }
        v24 = (_QWORD *)a1[4];
        if ( v21 == v20 )
        {
          a1[5] = 0;
          v30 = &v24[2 * (unsigned int)v21];
          do
          {
            if ( v24 )
              *v24 = -4096;
            v24 += 2;
          }
          while ( v30 != v24 );
        }
        else
        {
          sub_C7D6A0(v24, v6, 8);
          v25 = ((((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                   | (4 * v21 / 3u + 1)
                   | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                 | (4 * v21 / 3u + 1)
                 | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                 | (4 * v21 / 3u + 1)
                 | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
               | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
               | (4 * v21 / 3u + 1)
               | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 16;
          v26 = (v25
               | (((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                   | (4 * v21 / 3u + 1)
                   | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                 | (4 * v21 / 3u + 1)
                 | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
                 | (4 * v21 / 3u + 1)
                 | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
               | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
               | (4 * v21 / 3u + 1)
               | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1))
              + 1;
          *((_DWORD *)a1 + 12) = v26;
          v27 = (_QWORD *)sub_C7D670(16 * v26, 8);
          v28 = *((unsigned int *)a1 + 12);
          a1[5] = 0;
          a1[4] = (__int64)v27;
          for ( i = &v27[2 * v28]; i != v27; v27 += 2 )
          {
            if ( v27 )
              *v27 = -4096;
          }
        }
        goto LABEL_15;
      }
      if ( v20 )
      {
        sub_C7D6A0(a1[4], v6, 8);
        a1[4] = 0;
        a1[5] = 0;
        *((_DWORD *)a1 + 12) = 0;
        goto LABEL_15;
      }
    }
    a1[5] = 0;
  }
LABEL_15:
  sub_C9F930(a1 + 7);
  v9 = *((unsigned int *)a1 + 12);
  if ( (_DWORD)v9 )
  {
    v10 = (_QWORD *)a1[4];
    v11 = &v10[2 * v9];
    do
    {
      if ( *v10 != -8192 && *v10 != -4096 )
      {
        v12 = v10[1];
        if ( v12 )
        {
          sub_C9F8C0(v10[1]);
          j_j___libc_free_0(v12, 176);
        }
      }
      v10 += 2;
    }
    while ( v11 != v10 );
    v9 = *((unsigned int *)a1 + 12);
  }
  v13 = 16 * v9;
  sub_C7D6A0(a1[4], v13, 8);
  if ( *((_DWORD *)a1 + 3) )
  {
    v14 = *((unsigned int *)a1 + 2);
    v15 = *a1;
    if ( (_DWORD)v14 )
    {
      v16 = 8 * v14;
      v17 = 0;
      do
      {
        v18 = *(_QWORD **)(v15 + v17);
        if ( v18 != (_QWORD *)-8LL )
        {
          if ( v18 )
          {
            v13 = *v18 + 17LL;
            sub_C7D6A0(v18, v13, 8);
            v15 = *a1;
          }
        }
        v17 += 8;
      }
      while ( v17 != v16 );
    }
  }
  else
  {
    v15 = *a1;
  }
  _libc_free(v15, v13);
  j_j___libc_free_0(a1, 168);
}
