// Function: sub_26CAC90
// Address: 0x26cac90
//
__int64 __fastcall sub_26CAC90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r10d
  __int64 *v9; // r12
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r9
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-40h] BYREF
  char v17; // [rsp+24h] [rbp-2Ch]

  if ( unk_4F838D4 )
  {
    sub_3143F80(v16, a2, a3);
    result = 0;
    if ( !v17 )
      return result;
  }
  v5 = sub_B10CD0(a2 + 48);
  v15 = v5;
  if ( !v5 )
    return *(_QWORD *)(a1 + 1200);
  v6 = *(_DWORD *)(a1 + 32);
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = 1;
    v9 = 0;
    v10 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v5 == *v11 )
      return v11[1];
    while ( v12 != -4096 )
    {
      if ( !v9 && v12 == -8192 )
        v9 = v11;
      v10 = (v6 - 1) & (v8 + v10);
      v11 = (__int64 *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
        return v11[1];
      ++v8;
    }
    v14 = *(_DWORD *)(a1 + 24);
    if ( !v9 )
      v9 = v11;
    ++*(_QWORD *)(a1 + 8);
    v13 = v14 + 1;
    v16[0] = v9;
    if ( 4 * v13 < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 28) - v13 > v6 >> 3 )
        goto LABEL_12;
      goto LABEL_11;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 8);
    v16[0] = 0;
  }
  v6 *= 2;
LABEL_11:
  sub_26CAAB0(a1 + 8, v6);
  sub_26C3450(a1 + 8, &v15, v16);
  v5 = v15;
  v9 = (__int64 *)v16[0];
  v13 = *(_DWORD *)(a1 + 24) + 1;
LABEL_12:
  *(_DWORD *)(a1 + 24) = v13;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v9 = v5;
  v9[1] = 0;
  if ( unk_4F838D3 )
    result = sub_317EC90(*(_QWORD *)(a1 + 1512), v15);
  else
    result = sub_C1C070(*(_QWORD *)(a1 + 1200), v15, *(_QWORD *)(*(_QWORD *)(a1 + 1136) + 88LL), (_QWORD *)(a1 + 1352));
  v9[1] = result;
  return result;
}
