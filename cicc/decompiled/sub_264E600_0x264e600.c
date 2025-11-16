// Function: sub_264E600
// Address: 0x264e600
//
__int64 __fastcall sub_264E600(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdx
  size_t v4; // rdx
  void *v5; // rdi
  unsigned int v6; // ecx
  unsigned int v7; // eax
  _DWORD *v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !(_DWORD)result )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 <= 0x40 )
      goto LABEL_4;
    result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 4 * v3, 4);
    *(_DWORD *)(a1 + 24) = 0;
LABEL_21:
    *(_QWORD *)(a1 + 8) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
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
    goto LABEL_6;
  }
  v7 = result - 1;
  if ( v7 )
  {
    _BitScanReverse(&v7, v7);
    v8 = *(_DWORD **)(a1 + 8);
    v9 = (unsigned int)(1 << (33 - (v7 ^ 0x1F)));
    if ( (int)v9 < 64 )
      v9 = 64;
    if ( (_DWORD)v9 == (_DWORD)v3 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v8[v9];
      do
      {
        if ( v8 )
          *v8 = -1;
        ++v8;
      }
      while ( (_DWORD *)result != v8 );
      return result;
    }
  }
  else
  {
    v8 = *(_DWORD **)(a1 + 8);
    LODWORD(v9) = 64;
  }
  sub_C7D6A0((__int64)v8, 4 * v3, 4);
  result = sub_AF1560(4 * (int)v9 / 3u + 1);
  *(_DWORD *)(a1 + 24) = result;
  if ( !(_DWORD)result )
    goto LABEL_21;
  result = sub_C7D670(4LL * (unsigned int)result, 4);
  v10 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = result;
  for ( i = result + 4 * v10; i != result; result += 4 )
  {
    if ( result )
      *(_DWORD *)result = -1;
  }
  return result;
}
