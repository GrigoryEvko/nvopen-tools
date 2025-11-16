// Function: sub_3504DF0
// Address: 0x3504df0
//
void __fastcall sub_3504DF0(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  int v7; // r15d
  unsigned int v8; // eax
  _QWORD *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r14
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // r8
  int v15; // edx
  int v16; // ebx
  unsigned int v17; // r15d
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *i; // rdx
  _QWORD *v25; // rax
  unsigned __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = a1 + 8;
  *(_QWORD *)(v2 - 8) = 0;
  *(_QWORD *)(v2 + 216) = 0;
  sub_2DFA900(v2);
  sub_2DFA900(a1 + 120);
  v3 = *(_QWORD **)(a1 + 80);
  while ( v3 )
  {
    v4 = (unsigned __int64)v3;
    v3 = (_QWORD *)*v3;
    v5 = *(_QWORD *)(v4 + 104);
    if ( v5 != v4 + 120 )
      _libc_free(v5);
    v6 = *(_QWORD *)(v4 + 56);
    if ( v6 != v4 + 72 )
      _libc_free(v6);
    j_j___libc_free_0(v4);
  }
  memset(*(void **)(a1 + 64), 0, 8LL * *(_QWORD *)(a1 + 72));
  v7 = *(_DWORD *)(a1 + 248);
  ++*(_QWORD *)(a1 + 232);
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  if ( v7 || *(_DWORD *)(a1 + 252) )
  {
    v8 = 4 * v7;
    v9 = *(_QWORD **)(a1 + 240);
    v10 = *(unsigned int *)(a1 + 256);
    v11 = 16 * v10;
    if ( (unsigned int)(4 * v7) < 0x40 )
      v8 = 64;
    v12 = &v9[(unsigned __int64)v11 / 8];
    if ( (unsigned int)v10 <= v8 )
    {
      for ( ; v9 != v12; v9 += 2 )
      {
        if ( *v9 != -4096 )
        {
          if ( *v9 != -8192 )
          {
            v13 = v9[1];
            if ( v13 )
            {
              if ( !*(_BYTE *)(v13 + 28) )
                _libc_free(*(_QWORD *)(v13 + 8));
              j_j___libc_free_0(v13);
            }
          }
          *v9 = -4096;
        }
      }
      goto LABEL_21;
    }
    while ( 1 )
    {
      while ( *v9 == -8192 )
      {
LABEL_27:
        v9 += 2;
        if ( v12 == v9 )
          goto LABEL_31;
      }
      if ( *v9 != -4096 )
      {
        v14 = v9[1];
        if ( v14 )
        {
          if ( !*(_BYTE *)(v14 + 28) )
          {
            v26 = v9[1];
            _libc_free(*(_QWORD *)(v14 + 8));
            v14 = v26;
          }
          j_j___libc_free_0(v14);
        }
        goto LABEL_27;
      }
      v9 += 2;
      if ( v12 == v9 )
      {
LABEL_31:
        v15 = *(_DWORD *)(a1 + 256);
        if ( v7 )
        {
          v16 = 64;
          v17 = v7 - 1;
          if ( v17 )
          {
            _BitScanReverse(&v18, v17);
            v16 = 1 << (33 - (v18 ^ 0x1F));
            if ( v16 < 64 )
              v16 = 64;
          }
          v19 = *(_QWORD **)(a1 + 240);
          if ( v16 == v15 )
          {
            *(_QWORD *)(a1 + 248) = 0;
            v25 = &v19[2 * (unsigned int)v16];
            do
            {
              if ( v19 )
                *v19 = -4096;
              v19 += 2;
            }
            while ( v25 != v19 );
          }
          else
          {
            sub_C7D6A0((__int64)v19, v11, 8);
            v20 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                     | (4 * v16 / 3u + 1)
                     | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                   | (4 * v16 / 3u + 1)
                   | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                   | (4 * v16 / 3u + 1)
                   | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                 | (4 * v16 / 3u + 1)
                 | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
            v21 = (v20
                 | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                     | (4 * v16 / 3u + 1)
                     | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                   | (4 * v16 / 3u + 1)
                   | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                   | (4 * v16 / 3u + 1)
                   | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
                 | (4 * v16 / 3u + 1)
                 | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 256) = v21;
            v22 = (_QWORD *)sub_C7D670(16 * v21, 8);
            v23 = *(unsigned int *)(a1 + 256);
            *(_QWORD *)(a1 + 248) = 0;
            *(_QWORD *)(a1 + 240) = v22;
            for ( i = &v22[2 * v23]; i != v22; v22 += 2 )
            {
              if ( v22 )
                *v22 = -4096;
            }
          }
          return;
        }
        if ( v15 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 240), v11, 8);
          *(_QWORD *)(a1 + 240) = 0;
          *(_QWORD *)(a1 + 248) = 0;
          *(_DWORD *)(a1 + 256) = 0;
          return;
        }
LABEL_21:
        *(_QWORD *)(a1 + 248) = 0;
        return;
      }
    }
  }
}
