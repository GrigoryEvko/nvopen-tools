// Function: sub_21FA2D0
// Address: 0x21fa2d0
//
__int64 __fastcall sub_21FA2D0(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  unsigned int v6; // ecx
  _DWORD *v7; // rdi
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 j; // rdx

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_4;
  }
  v6 = 4 * v2;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v4 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 8);
    for ( i = result + 16 * v4; result != i; result += 16 )
      *(_DWORD *)result = -1;
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v7 = *(_DWORD **)(a1 + 8);
  v8 = v2 - 1;
  if ( !v8 )
  {
    v13 = 2048;
    v12 = 128;
LABEL_16:
    j___libc_free_0(v7);
    *(_DWORD *)(a1 + 24) = v12;
    result = sub_22077B0(v13);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( j = result + 16 * v14; j != result; result += 16 )
    {
      if ( result )
        *(_DWORD *)result = -1;
    }
    return result;
  }
  _BitScanReverse(&v8, v8);
  v9 = 1 << (33 - (v8 ^ 0x1F));
  if ( v9 < 64 )
    v9 = 64;
  if ( (_DWORD)v4 != v9 )
  {
    v10 = (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
        | (4 * v9 / 3u + 1)
        | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)
        | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
          | (4 * v9 / 3u + 1)
          | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4);
    v11 = (v10 >> 8) | v10;
    v12 = (v11 | (v11 >> 16)) + 1;
    v13 = 16 * ((v11 | (v11 >> 16)) + 1);
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v7[4 * (unsigned int)v4];
  do
  {
    if ( v7 )
      *v7 = -1;
    v7 += 4;
  }
  while ( (_DWORD *)result != v7 );
  return result;
}
