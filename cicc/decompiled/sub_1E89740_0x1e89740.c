// Function: sub_1E89740
// Address: 0x1e89740
//
__int64 __fastcall sub_1E89740(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  size_t v4; // rdx
  void *v5; // rdi
  unsigned int v6; // ecx
  _DWORD *v7; // rdi
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !(_DWORD)result )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    goto LABEL_4;
  }
  v6 = 4 * result;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * result) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = 4 * v3;
    v5 = *(void **)(a1 + 8);
    if ( v4 )
      result = (__int64)memset(v5, 255, v4);
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v7 = *(_DWORD **)(a1 + 8);
  v8 = result - 1;
  if ( !v8 )
  {
    v13 = 512;
    v12 = 128;
LABEL_16:
    j___libc_free_0(v7);
    *(_DWORD *)(a1 + 24) = v12;
    result = sub_22077B0(v13);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = result;
    for ( i = result + 4 * v14; i != result; result += 4 )
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
  if ( (_DWORD)v3 != v9 )
  {
    v10 = (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
        | (4 * v9 / 3u + 1)
        | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)
        | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
          | (4 * v9 / 3u + 1)
          | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4);
    v11 = (v10 >> 8) | v10;
    v12 = (v11 | (v11 >> 16)) + 1;
    v13 = 4 * ((v11 | (v11 >> 16)) + 1);
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (__int64)&v7[v3];
  do
  {
    if ( v7 )
      *v7 = -1;
    ++v7;
  }
  while ( (_DWORD *)result != v7 );
  return result;
}
