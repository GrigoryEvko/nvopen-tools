// Function: sub_2051350
// Address: 0x2051350
//
void __fastcall sub_2051350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // rdx
  _QWORD *v11; // r13
  unsigned int v12; // eax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // rbx
  unsigned int v17; // r14d
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *i; // rdx
  _QWORD *v25; // rax

  v5 = *(_QWORD *)(a1 + 552);
  *(_QWORD *)(a1 + 568) = a3;
  *(_QWORD *)(a1 + 720) = a2;
  *(_QWORD *)(a1 + 576) = a4;
  v6 = sub_1E0A0C0(*(_QWORD *)(v5 + 32));
  v7 = *(_DWORD *)(a1 + 744);
  *(_QWORD *)(a1 + 560) = v6;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 48LL);
  ++*(_QWORD *)(a1 + 728);
  *(_QWORD *)(a1 + 768) = v8;
  if ( v7 || *(_DWORD *)(a1 + 748) )
  {
    v9 = *(_QWORD **)(a1 + 736);
    v10 = *(unsigned int *)(a1 + 752);
    v11 = &v9[5 * v10];
    v12 = 4 * v7;
    if ( (unsigned int)(4 * v7) < 0x40 )
      v12 = 64;
    if ( (unsigned int)v10 <= v12 )
    {
      for ( ; v9 != v11; v9 += 5 )
      {
        if ( *v9 != -8 )
        {
          if ( *v9 != -16 )
          {
            v13 = v9[1];
            if ( (_QWORD *)v13 != v9 + 3 )
              _libc_free(v13);
          }
          *v9 = -8;
        }
      }
LABEL_13:
      *(_QWORD *)(a1 + 744) = 0;
      return;
    }
    do
    {
      if ( *v9 != -16 && *v9 != -8 )
      {
        v14 = v9[1];
        if ( (_QWORD *)v14 != v9 + 3 )
          _libc_free(v14);
      }
      v9 += 5;
    }
    while ( v9 != v11 );
    v15 = *(_DWORD *)(a1 + 752);
    if ( !v7 )
    {
      if ( v15 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 736));
        *(_QWORD *)(a1 + 736) = 0;
        *(_QWORD *)(a1 + 744) = 0;
        *(_DWORD *)(a1 + 752) = 0;
        return;
      }
      goto LABEL_13;
    }
    v16 = 64;
    v17 = v7 - 1;
    if ( v17 )
    {
      _BitScanReverse(&v18, v17);
      v16 = (unsigned int)(1 << (33 - (v18 ^ 0x1F)));
      if ( (int)v16 < 64 )
        v16 = 64;
    }
    v19 = *(_QWORD **)(a1 + 736);
    if ( (_DWORD)v16 == v15 )
    {
      *(_QWORD *)(a1 + 744) = 0;
      v25 = &v19[5 * v16];
      do
      {
        if ( v19 )
          *v19 = -8;
        v19 += 5;
      }
      while ( v25 != v19 );
    }
    else
    {
      j___libc_free_0(v19);
      v20 = ((((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v16 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 16;
      v21 = (v20
           | (((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v16 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 752) = v21;
      v22 = (_QWORD *)sub_22077B0(40 * v21);
      v23 = *(unsigned int *)(a1 + 752);
      *(_QWORD *)(a1 + 744) = 0;
      *(_QWORD *)(a1 + 736) = v22;
      for ( i = &v22[5 * v23]; i != v22; v22 += 5 )
      {
        if ( v22 )
          *v22 = -8;
      }
    }
  }
}
