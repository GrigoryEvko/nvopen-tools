// Function: sub_1377190
// Address: 0x1377190
//
__int64 __fastcall sub_1377190(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  unsigned int v6; // ecx
  __int64 v7; // rdi
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 j; // rdx

  v2 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 52);
    if ( !(_DWORD)result )
      return result;
    v4 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 40));
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_DWORD *)(a1 + 56) = 0;
      return result;
    }
    goto LABEL_4;
  }
  v6 = 4 * v2;
  v4 = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v4 <= v6 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 40);
    for ( i = result + 24 * v4; i != result; *(_DWORD *)(result - 16) = -1 )
    {
      *(_QWORD *)result = -8;
      result += 24;
    }
    *(_QWORD *)(a1 + 48) = 0;
    return result;
  }
  v7 = *(_QWORD *)(a1 + 40);
  v8 = v2 - 1;
  if ( !v8 )
  {
    v13 = 3072;
    v12 = 128;
LABEL_16:
    j___libc_free_0(v7);
    *(_DWORD *)(a1 + 56) = v12;
    result = sub_22077B0(v13);
    v14 = *(unsigned int *)(a1 + 56);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 40) = result;
    for ( j = result + 24 * v14; j != result; result += 24 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    return result;
  }
  _BitScanReverse(&v8, v8);
  v9 = 1 << (33 - (v8 ^ 0x1F));
  if ( v9 < 64 )
    v9 = 64;
  if ( (_DWORD)v4 != v9 )
  {
    v10 = (4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1);
    v11 = ((((v10 >> 2) | v10 | (((v10 >> 2) | v10) >> 4)) >> 8)
         | (v10 >> 2)
         | v10
         | (((v10 >> 2) | v10) >> 4)
         | (((((v10 >> 2) | v10 | (((v10 >> 2) | v10) >> 4)) >> 8) | (v10 >> 2) | v10 | (((v10 >> 2) | v10) >> 4)) >> 16))
        + 1;
    v12 = v11;
    v13 = 24 * v11;
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 48) = 0;
  result = v7 + 24 * v4;
  do
  {
    if ( v7 )
    {
      *(_QWORD *)v7 = -8;
      *(_DWORD *)(v7 + 8) = -1;
    }
    v7 += 24;
  }
  while ( result != v7 );
  return result;
}
