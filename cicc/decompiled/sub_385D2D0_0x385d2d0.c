// Function: sub_385D2D0
// Address: 0x385d2d0
//
void __fastcall sub_385D2D0(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r12
  unsigned __int64 *v7; // rdi
  unsigned __int64 *v8; // rdi
  int v9; // edx
  int v10; // ebx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  _QWORD *v19; // rax

  v2 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( v2 || *(_DWORD *)(a1 + 180) )
  {
    v3 = *(_QWORD **)(a1 + 168);
    v4 = 4 * v2;
    v5 = *(unsigned int *)(a1 + 184);
    v6 = &v3[2 * v5];
    if ( (unsigned int)(4 * v2) < 0x40 )
      v4 = 64;
    if ( (unsigned int)v5 <= v4 )
    {
      for ( ; v3 != v6; v3 += 2 )
      {
        if ( *v3 != -8 )
        {
          if ( *v3 != -16 )
          {
            v7 = (unsigned __int64 *)v3[1];
            if ( v7 )
              sub_385CEA0(v7);
          }
          *v3 = -8;
        }
      }
LABEL_13:
      *(_QWORD *)(a1 + 176) = 0;
      return;
    }
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v8 = (unsigned __int64 *)v3[1];
        if ( v8 )
          sub_385CEA0(v8);
      }
      v3 += 2;
    }
    while ( v3 != v6 );
    v9 = *(_DWORD *)(a1 + 184);
    if ( !v2 )
    {
      if ( v9 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 168));
        *(_QWORD *)(a1 + 168) = 0;
        *(_QWORD *)(a1 + 176) = 0;
        *(_DWORD *)(a1 + 184) = 0;
        return;
      }
      goto LABEL_13;
    }
    v10 = 64;
    v11 = v2 - 1;
    if ( v11 )
    {
      _BitScanReverse(&v12, v11);
      v10 = 1 << (33 - (v12 ^ 0x1F));
      if ( v10 < 64 )
        v10 = 64;
    }
    v13 = *(_QWORD **)(a1 + 168);
    if ( v10 == v9 )
    {
      *(_QWORD *)(a1 + 176) = 0;
      v19 = &v13[2 * (unsigned int)v10];
      do
      {
        if ( v13 )
          *v13 = -8;
        v13 += 2;
      }
      while ( v19 != v13 );
    }
    else
    {
      j___libc_free_0((unsigned __int64)v13);
      v14 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
               | (4 * v10 / 3u + 1)
               | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
             | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
      v15 = (v14
           | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
               | (4 * v10 / 3u + 1)
               | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
             | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 184) = v15;
      v16 = (_QWORD *)sub_22077B0(16 * v15);
      v17 = *(unsigned int *)(a1 + 184);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = v16;
      for ( i = &v16[2 * v17]; i != v16; v16 += 2 )
      {
        if ( v16 )
          *v16 = -8;
      }
    }
  }
}
