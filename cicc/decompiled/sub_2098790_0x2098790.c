// Function: sub_2098790
// Address: 0x2098790
//
__int64 __fastcall sub_2098790(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  unsigned __int64 *v6; // r12
  unsigned int v7; // ecx
  __int64 v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // r13d
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 j; // rdx

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v2;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v4 <= v7 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 8);
    for ( i = result + 32 * v4; i != result; *(_DWORD *)(result - 24) = -1 )
    {
      *(_QWORD *)result = 0;
      result += 32;
    }
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v2 - 1;
  if ( !v9 )
  {
    v14 = 4096;
    v13 = 128;
LABEL_19:
    j___libc_free_0(v8);
    *(_DWORD *)(a1 + 24) = v13;
    result = sub_22077B0(v14);
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( j = result + 32 * v15; j != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v9, v9);
  v10 = (unsigned int)(1 << (33 - (v9 ^ 0x1F)));
  if ( (int)v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v10 != (_DWORD)v4 )
  {
    v11 = (((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v10 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)
        | (((((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v10 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 4);
    v12 = (v11 >> 8) | v11;
    v13 = (v12 | (v12 >> 16)) + 1;
    v14 = 32 * ((v12 | (v12 >> 16)) + 1);
    goto LABEL_19;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = v8 + 32 * v10;
  do
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = 0;
      *(_DWORD *)(v8 + 8) = -1;
    }
    v8 += 32;
  }
  while ( result != v8 );
LABEL_7:
  v6 = *(unsigned __int64 **)(a1 + 32);
  if ( ((unsigned __int8)v6 & 1) == 0 && v6 )
  {
    _libc_free(*v6);
    result = j_j___libc_free_0(v6, 24);
  }
  *(_QWORD *)(a1 + 32) = 1;
  return result;
}
