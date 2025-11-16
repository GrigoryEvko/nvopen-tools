// Function: sub_1629450
// Address: 0x1629450
//
__int64 *__fastcall sub_1629450(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  int v5; // r12d
  __int64 *result; // rax
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // r12d
  __int64 *v10; // rcx
  __int64 v11; // rdx
  int v12; // r9d
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v16[4]; // [rsp+10h] [rbp-50h] BYREF
  int v17; // [rsp+30h] [rbp-30h]

  v3 = a1;
  v16[0] = 0;
  v16[1] = 0;
  v4 = *(unsigned int *)(a1 + 8);
  v15 = a1;
  v4 *= -8;
  v16[2] = (__int64 *)(a1 + v4);
  v16[3] = (__int64 *)(-v4 >> 3);
  v5 = *(_DWORD *)(a1 + 4);
  v17 = v5;
  result = (__int64 *)sub_161C9B0(a2, (__int64)v16);
  if ( result )
    return result;
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    v9 = (v7 - 1) & v5;
    v10 = (__int64 *)(v8 + 8LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
      return (__int64 *)v3;
    v12 = 1;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || result )
        v10 = result;
      v9 = (v7 - 1) & (v12 + v9);
      v11 = *(_QWORD *)(v8 + 8LL * v9);
      if ( v3 == v11 )
        return (__int64 *)v3;
      ++v12;
      result = v10;
      v10 = (__int64 *)(v8 + 8LL * v9);
    }
    if ( !result )
      result = v10;
    v14 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v13 = v14 + 1;
    if ( 4 * v13 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a2 + 20) - v13 > v7 >> 3 )
        goto LABEL_13;
      goto LABEL_12;
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  v7 *= 2;
LABEL_12:
  sub_1627160(a2, v7);
  sub_1621680(a2, &v15, v16);
  result = v16[0];
  v3 = v15;
  v13 = *(_DWORD *)(a2 + 16) + 1;
LABEL_13:
  *(_DWORD *)(a2 + 16) = v13;
  if ( *result != -8 )
    --*(_DWORD *)(a2 + 20);
  *result = v3;
  return (__int64 *)v15;
}
