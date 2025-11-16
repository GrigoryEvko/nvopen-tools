// Function: sub_3373460
// Address: 0x3373460
//
void __fastcall sub_3373460(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *j; // rdx
  __int64 v6; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // r12
  unsigned __int64 v10; // rbx
  __int64 v11; // rsi
  unsigned int v12; // ecx
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  int v15; // ebx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  _QWORD *v21; // rax

  v2 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( v2 )
  {
    v12 = 4 * v2;
    v3 = *(unsigned int *)(a1 + 96);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v12 = 64;
    if ( v12 >= (unsigned int)v3 )
      goto LABEL_4;
    v13 = v2 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v13, v13);
      v14 = *(_QWORD **)(a1 + 80);
      v15 = 1 << (33 - (v13 ^ 0x1F));
      if ( v15 < 64 )
        v15 = 64;
      if ( (_DWORD)v3 == v15 )
      {
        *(_QWORD *)(a1 + 88) = 0;
        v21 = &v14[2 * (unsigned int)v3];
        do
        {
          if ( v14 )
            *v14 = -4096;
          v14 += 2;
        }
        while ( v21 != v14 );
        goto LABEL_7;
      }
    }
    else
    {
      v14 = *(_QWORD **)(a1 + 80);
      v15 = 64;
    }
    sub_C7D6A0((__int64)v14, 16 * v3, 8);
    v16 = ((((((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
             | (4 * v15 / 3u + 1)
             | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
           | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
           | (4 * v15 / 3u + 1)
           | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
           | (4 * v15 / 3u + 1)
           | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
         | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
         | (4 * v15 / 3u + 1)
         | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 16;
    v17 = (v16
         | (((((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
             | (4 * v15 / 3u + 1)
             | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
           | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
           | (4 * v15 / 3u + 1)
           | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
           | (4 * v15 / 3u + 1)
           | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4)
         | (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
         | (4 * v15 / 3u + 1)
         | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 96) = v17;
    v18 = (_QWORD *)sub_C7D670(16 * v17, 8);
    v19 = *(unsigned int *)(a1 + 96);
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 80) = v18;
    for ( i = &v18[2 * v19]; i != v18; v18 += 2 )
    {
      if ( v18 )
        *v18 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 92) )
  {
    v3 = *(unsigned int *)(a1 + 96);
    if ( (unsigned int)v3 <= 0x40 )
    {
LABEL_4:
      v4 = *(_QWORD **)(a1 + 80);
      for ( j = &v4[2 * v3]; j != v4; v4 += 2 )
        *v4 = -4096;
      *(_QWORD *)(a1 + 88) = 0;
      goto LABEL_7;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 80), 16 * v3, 8);
    *(_QWORD *)(a1 + 80) = 0;
    *(_QWORD *)(a1 + 88) = 0;
    *(_DWORD *)(a1 + 96) = 0;
  }
LABEL_7:
  v6 = *(_QWORD *)(a1 + 104);
  v7 = v6 + 32LL * *(unsigned int *)(a1 + 112);
  while ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 - 24);
      v9 = *(_QWORD *)(v7 - 16);
      v7 -= 32;
      v10 = v8;
      if ( v9 != v8 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 24);
          if ( v11 )
            sub_B91220(v10 + 24, v11);
          v10 += 32LL;
        }
        while ( v9 != v10 );
        v8 = *(_QWORD *)(v7 + 8);
      }
      if ( !v8 )
        break;
      j_j___libc_free_0(v8);
      if ( v6 == v7 )
        goto LABEL_16;
    }
  }
LABEL_16:
  *(_DWORD *)(a1 + 112) = 0;
}
