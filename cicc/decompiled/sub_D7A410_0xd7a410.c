// Function: sub_D7A410
// Address: 0xd7a410
//
void __fastcall sub_D7A410(__int64 a1, __int64 a2, size_t a3)
{
  int v4; // ebx
  size_t v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rdi
  char *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // r15
  unsigned int v15; // ebx
  unsigned int v16; // eax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __m128i v21; // [rsp+0h] [rbp-60h] BYREF
  char *v22[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v23; // [rsp+20h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v4 )
  {
    v5 = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)v5 )
      return;
    v6 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v6 > 0x40 )
    {
      sub_D78B00((_QWORD *)a1, a2, v5);
      if ( *(_DWORD *)(a1 + 24) )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v6, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_17;
    }
LABEL_4:
    v7 = *(_QWORD *)(a1 + 8);
    v21.m128i_i64[0] = 0;
    v21.m128i_i64[1] = -1;
    v8 = v7 + 40 * v6;
    v22[0] = 0;
    v22[1] = 0;
    v23 = 0;
    if ( v8 != v7 )
    {
      do
      {
        v9 = v7 + 16;
        v7 += 40;
        *(__m128i *)(v7 - 40) = _mm_loadu_si128(&v21);
        sub_D76A50(v9, v22);
      }
      while ( v7 != v8 );
      v10 = v22[0];
      v11 = v23;
      *(_QWORD *)(a1 + 16) = 0;
      v12 = v11 - (_QWORD)v10;
      if ( v10 )
        j_j___libc_free_0(v10, v12);
      return;
    }
LABEL_17:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v13 = 4 * v4;
  v6 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v6 <= v13 )
    goto LABEL_4;
  v14 = 64;
  sub_D78B00((_QWORD *)a1, a2, a3);
  v15 = v4 - 1;
  if ( v15 )
  {
    _BitScanReverse(&v16, v15);
    v14 = (unsigned int)(1 << (33 - (v16 ^ 0x1F)));
    if ( (int)v14 < 64 )
      v14 = 64;
  }
  if ( *(_DWORD *)(a1 + 24) == (_DWORD)v14 )
  {
    v19 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)(a1 + 16) = 0;
    v20 = &v19[5 * v14];
    do
    {
      if ( v19 )
      {
        *v19 = 0;
        v19[1] = -1;
        v19[2] = 0;
        v19[3] = 0;
        v19[4] = 0;
      }
      v19 += 5;
    }
    while ( v20 != v19 );
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v6, 8);
    v17 = ((((((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v14 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 16;
    v18 = (v17
         | (((((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v14 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v14 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v14 / 3u + 1) | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v14 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v14 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v18;
    *(_QWORD *)(a1 + 8) = sub_C7D670(40 * v18, 8);
    sub_D7A3B0(a1);
  }
}
