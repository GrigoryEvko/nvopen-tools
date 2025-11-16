// Function: sub_190D6F0
// Address: 0x190d6f0
//
void __fastcall sub_190D6F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // r12d
  __int64 v8; // rbx
  __int64 v9; // rcx
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r13
  _BYTE *v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 v15; // rdx
  int v16; // ebx
  unsigned int v17; // r12d
  unsigned int v18; // eax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // [rsp+18h] [rbp-A8h]
  void *s2[2]; // [rsp+28h] [rbp-98h] BYREF
  _BYTE v23[24]; // [rsp+38h] [rbp-88h] BYREF
  int v24; // [rsp+50h] [rbp-70h]
  char v25; // [rsp+60h] [rbp-60h]
  _BYTE *v26; // [rsp+68h] [rbp-58h]
  __int64 v27; // [rsp+70h] [rbp-50h]
  _BYTE v28[72]; // [rsp+78h] [rbp-48h] BYREF

  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v7 || *(_DWORD *)(a1 + 20) )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = 64;
    v10 = 4 * v7;
    v11 = *(unsigned int *)(a1 + 24);
    v12 = v8 + (v11 << 6);
    if ( (unsigned int)(4 * v7) < 0x40 )
      v10 = 64;
    if ( v10 < (unsigned int)v11 )
    {
      do
      {
        v14 = *(_QWORD *)(v8 + 24);
        if ( v14 != v8 + 40 )
          _libc_free(v14);
        v8 += 64;
      }
      while ( v8 != v12 );
      v15 = *(unsigned int *)(a1 + 24);
      if ( v7 )
      {
        v16 = 64;
        v17 = v7 - 1;
        if ( v17 )
        {
          _BitScanReverse(&v18, v17);
          v9 = 33 - (v18 ^ 0x1F);
          v16 = 1 << (33 - (v18 ^ 0x1F));
          if ( v16 < 64 )
            v16 = 64;
        }
        if ( (_DWORD)v15 != v16 )
        {
          j___libc_free_0(*(_QWORD *)(a1 + 8));
          v19 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
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
          v20 = (v19
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
          *(_DWORD *)(a1 + 24) = v20;
          *(_QWORD *)(a1 + 8) = sub_22077B0(v20 << 6);
        }
      }
      else if ( (_DWORD)v15 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      sub_190D620(a1, a2, v15, v9, a5, a6);
      return;
    }
    v13 = v28;
    v25 = 0;
    s2[0] = v23;
    s2[1] = (void *)0x400000000LL;
    v24 = -2;
    v26 = v28;
    v27 = 0x400000000LL;
    if ( v8 == v12 )
    {
      *(_QWORD *)(a1 + 16) = 0;
    }
    else
    {
      do
      {
        if ( *(_DWORD *)v8 != -1 )
        {
          *(_DWORD *)v8 = -1;
          *(_QWORD *)(v8 + 8) = v21;
          *(_BYTE *)(v8 + 16) = 0;
          sub_1909410(v8 + 24, (__int64)s2, (__int64)v13, v9, a5, a6);
        }
        v8 += 64;
      }
      while ( v8 != v12 );
      *(_QWORD *)(a1 + 16) = 0;
      if ( s2[0] != v23 )
        _libc_free((unsigned __int64)s2[0]);
    }
  }
}
