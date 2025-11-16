// Function: sub_2510BF0
// Address: 0x2510bf0
//
unsigned __int64 __fastcall sub_2510BF0(__int64 a1)
{
  int v1; // eax
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  unsigned __int64 i; // rdx
  unsigned int v6; // ecx
  unsigned int v7; // eax
  _QWORD *v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rdx
  unsigned __int64 j; // rdx

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
    result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 8 * v4, 8);
    *(_DWORD *)(a1 + 24) = 0;
LABEL_21:
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
    for ( i = result + 8 * v4; result != i; result += 8LL )
      *(_QWORD *)result = -4;
    goto LABEL_6;
  }
  v7 = v1 - 1;
  if ( v7 )
  {
    _BitScanReverse(&v7, v7);
    v8 = *(_QWORD **)(a1 + 8);
    v9 = (unsigned int)(1 << (33 - (v7 ^ 0x1F)));
    if ( (int)v9 < 64 )
      v9 = 64;
    if ( (_DWORD)v9 == (_DWORD)v4 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (unsigned __int64)&v8[v9];
      do
      {
        if ( v8 )
          *v8 = -4;
        ++v8;
      }
      while ( (_QWORD *)result != v8 );
      return result;
    }
  }
  else
  {
    v8 = *(_QWORD **)(a1 + 8);
    LODWORD(v9) = 64;
  }
  sub_C7D6A0((__int64)v8, 8 * v4, 8);
  result = sub_2507810(v9);
  *(_DWORD *)(a1 + 24) = result;
  if ( !(_DWORD)result )
    goto LABEL_21;
  result = sub_C7D670(8LL * (unsigned int)result, 8);
  v10 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = result;
  for ( j = result + 8 * v10; j != result; result += 8LL )
  {
    if ( result )
      *(_QWORD *)result = -4;
  }
  return result;
}
