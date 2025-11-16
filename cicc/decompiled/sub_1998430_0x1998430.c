// Function: sub_1998430
// Address: 0x1998430
//
__int64 __fastcall sub_1998430(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r9d
  __int64 *v8; // r8
  unsigned int v9; // eax
  __int64 *v10; // rbx
  __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 result; // rax
  int v16; // eax
  int v17; // edx
  __int64 v18; // rax
  __int64 *v19; // [rsp+8h] [rbp-38h] BYREF
  __int64 v20; // [rsp+10h] [rbp-30h] BYREF
  __int64 v21; // [rsp+18h] [rbp-28h]

  v20 = a2;
  v5 = *(_DWORD *)(a1 + 24);
  v21 = 1;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_27:
    v5 *= 2;
    goto LABEL_28;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == a2 )
    goto LABEL_3;
  while ( v11 != -8 )
  {
    if ( !v8 && v11 == -16 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
      goto LABEL_3;
    ++v7;
  }
  v16 = *(_DWORD *)(a1 + 16);
  if ( v8 )
    v10 = v8;
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v5 )
    goto LABEL_27;
  v6 = a2;
  if ( v5 - *(_DWORD *)(a1 + 20) - v17 <= v5 >> 3 )
  {
LABEL_28:
    sub_1996920(a1, v5);
    sub_1992680(a1, &v20, &v19);
    v10 = v19;
    v6 = v20;
    v17 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v17;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v10 = v6;
  v10[1] = v21;
  v18 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, (int)v8, v7);
    v18 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v18) = a2;
  ++*(_DWORD *)(a1 + 40);
LABEL_3:
  v12 = v10[1];
  if ( (v12 & 1) != 0 )
    v13 = v12 >> 58;
  else
    v13 = *(unsigned int *)(v12 + 16);
  if ( a3 + 1 >= v13 )
    LODWORD(v13) = a3 + 1;
  sub_13A5100((unsigned __int64 *)v10 + 1, v13, 0, v6, (int)v8, v7);
  v14 = v10[1];
  if ( (v14 & 1) != 0 )
  {
    result = 2 * ((v14 >> 58 << 57) | ~(-1LL << (v14 >> 58)) & (~(-1LL << (v14 >> 58)) & (v14 >> 1) | (1LL << a3))) + 1;
    v10[1] = result;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)v14 + 8LL * ((unsigned int)a3 >> 6)) |= 1LL << a3;
    return 1LL << a3;
  }
  return result;
}
