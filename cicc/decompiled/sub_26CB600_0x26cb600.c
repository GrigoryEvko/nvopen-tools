// Function: sub_26CB600
// Address: 0x26cb600
//
__int64 __fastcall sub_26CB600(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // esi
  __int64 v5; // rdi
  int v6; // r10d
  __int64 *v7; // r12
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  __int64 result; // rax
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v16; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_B10CD0(a2 + 48);
  v15 = v3;
  if ( !v3 )
    return *(_QWORD *)(a1 + 1200);
  v4 = *(_DWORD *)(a1 + 32);
  if ( v4 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    v6 = 1;
    v7 = 0;
    v8 = (v4 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( v3 == *v9 )
      return v9[1];
    while ( v10 != -4096 )
    {
      if ( !v7 && v10 == -8192 )
        v7 = v9;
      v8 = (v4 - 1) & (v6 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( v3 == *v9 )
        return v9[1];
      ++v6;
    }
    v14 = *(_DWORD *)(a1 + 24);
    if ( !v7 )
      v7 = v9;
    ++*(_QWORD *)(a1 + 8);
    v12 = v14 + 1;
    v16 = v7;
    if ( 4 * v12 < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 28) - v12 > v4 >> 3 )
        goto LABEL_9;
      goto LABEL_8;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 8);
    v16 = 0;
  }
  v4 *= 2;
LABEL_8:
  sub_26CAAB0(a1 + 8, v4);
  sub_26C3450(a1 + 8, &v15, &v16);
  v3 = v15;
  v7 = v16;
  v12 = *(_DWORD *)(a1 + 24) + 1;
LABEL_9:
  *(_DWORD *)(a1 + 24) = v12;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v7 = v3;
  v13 = v15;
  v7[1] = 0;
  result = sub_C1C070(*(_QWORD *)(a1 + 1200), v13, *(_QWORD *)(*(_QWORD *)(a1 + 1136) + 88LL), 0);
  v7[1] = result;
  return result;
}
