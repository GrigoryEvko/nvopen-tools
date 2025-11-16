// Function: sub_1444070
// Address: 0x1444070
//
__int64 __fastcall sub_1444070(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  __int64 v6; // r12
  unsigned int v7; // ecx
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // r13d
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 j; // rdx

  v2 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 48));
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v2;
  v4 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v4 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 48);
    for ( i = result + 16 * v4; i != result; result += 16 )
      *(_QWORD *)result = -8;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_7;
  }
  v8 = *(_QWORD **)(a1 + 48);
  v9 = v2 - 1;
  if ( !v9 )
  {
    v14 = 2048;
    v13 = 128;
LABEL_18:
    j___libc_free_0(v8);
    *(_DWORD *)(a1 + 64) = v13;
    result = sub_22077B0(v14);
    v15 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = result;
    for ( j = result + 16 * v15; j != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v9, v9);
  v10 = 1 << (33 - (v9 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v4 != v10 )
  {
    v11 = (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
        | (4 * v10 / 3u + 1)
        | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)
        | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
          | (4 * v10 / 3u + 1)
          | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4);
    v12 = (v11 >> 8) | v11;
    v13 = (v12 | (v12 >> 16)) + 1;
    v14 = 16 * ((v12 | (v12 >> 16)) + 1);
    goto LABEL_18;
  }
  *(_QWORD *)(a1 + 56) = 0;
  result = (__int64)&v8[2 * (unsigned int)v4];
  do
  {
    if ( v8 )
      *v8 = -8;
    v8 += 2;
  }
  while ( (_QWORD *)result != v8 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 32);
  if ( v6 )
  {
    sub_1444060(*(_QWORD *)(a1 + 32));
    result = j_j___libc_free_0(v6, 112);
  }
  *(_QWORD *)(a1 + 32) = 0;
  return result;
}
