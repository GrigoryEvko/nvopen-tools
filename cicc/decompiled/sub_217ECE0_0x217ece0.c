// Function: sub_217ECE0
// Address: 0x217ece0
//
unsigned __int64 __fastcall sub_217ECE0(__int64 a1)
{
  int v1; // eax
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  unsigned __int64 i; // rdx
  unsigned int v6; // ecx
  _DWORD *v7; // rdi
  unsigned int v8; // eax
  int v9; // r13d

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
LABEL_18:
    *(_QWORD *)(a1 + 8) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 16) = 0;
    return result;
  }
  v6 = 4 * v1;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v4 <= v6 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 8);
    for ( i = result + 8 * v4; result != i; result += 8LL )
      *(_DWORD *)result = -1;
    goto LABEL_6;
  }
  v7 = *(_DWORD **)(a1 + 8);
  v8 = v1 - 1;
  if ( !v8 )
  {
    v9 = 64;
    goto LABEL_15;
  }
  _BitScanReverse(&v8, v8);
  v9 = 1 << (33 - (v8 ^ 0x1F));
  if ( v9 < 64 )
    v9 = 64;
  if ( (_DWORD)v4 != v9 )
  {
LABEL_15:
    j___libc_free_0(v7);
    result = sub_217D900(v9);
    *(_DWORD *)(a1 + 24) = result;
    if ( (_DWORD)result )
    {
      *(_QWORD *)(a1 + 8) = sub_22077B0(8LL * (unsigned int)result);
      return (unsigned __int64)sub_217ECA0(a1);
    }
    goto LABEL_18;
  }
  *(_QWORD *)(a1 + 16) = 0;
  result = (unsigned __int64)&v7[2 * v4];
  do
  {
    if ( v7 )
      *v7 = -1;
    v7 += 2;
  }
  while ( (_DWORD *)result != v7 );
  return result;
}
