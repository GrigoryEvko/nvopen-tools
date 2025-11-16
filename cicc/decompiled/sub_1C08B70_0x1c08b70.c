// Function: sub_1C08B70
// Address: 0x1c08b70
//
__int64 __fastcall sub_1C08B70(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  unsigned int v6; // ecx
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // r12
  __int64 v11; // rdx
  __int64 j; // rdx

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 <= 0x40 )
      goto LABEL_4;
    result = j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_DWORD *)(a1 + 24) = 0;
LABEL_23:
    *(_QWORD *)(a1 + 8) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v6 = 4 * v1;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v4 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 8);
    for ( i = result + 8 * v4; result != i; result += 8 )
      *(_QWORD *)result = -8;
    goto LABEL_6;
  }
  v7 = *(_QWORD **)(a1 + 8);
  v8 = v1 - 1;
  if ( v8 )
  {
    _BitScanReverse(&v8, v8);
    v9 = 1 << (33 - (v8 ^ 0x1F));
    if ( v9 < 64 )
      v9 = 64;
    if ( (_DWORD)v4 == v9 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v7[v4];
      do
      {
        if ( v7 )
          *v7 = -8;
        ++v7;
      }
      while ( (_QWORD *)result != v7 );
      return result;
    }
    v10 = 4 * v9 / 3u + 1;
  }
  else
  {
    v10 = 86;
  }
  j___libc_free_0(v7);
  result = sub_1454B60(v10);
  *(_DWORD *)(a1 + 24) = result;
  if ( !(_DWORD)result )
    goto LABEL_23;
  result = sub_22077B0(8LL * (unsigned int)result);
  v11 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = result;
  for ( j = result + 8 * v11; j != result; result += 8 )
  {
    if ( result )
      *(_QWORD *)result = -8;
  }
  return result;
}
