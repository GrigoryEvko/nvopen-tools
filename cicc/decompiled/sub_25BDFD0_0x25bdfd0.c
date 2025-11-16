// Function: sub_25BDFD0
// Address: 0x25bdfd0
//
__int64 __fastcall sub_25BDFD0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 *v11; // rdi
  __int64 v12; // r9
  int v13; // edi
  int v14; // r11d
  __int64 v15[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_B46790((unsigned __int8 *)a2, 1u);
  if ( !(_BYTE)v2 )
    return v2;
  if ( *(_BYTE *)a2 != 85 )
    return v2;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    return v2;
  v5 = *a1;
  v15[0] = *(_QWORD *)(a2 - 32);
  if ( !*(_DWORD *)(v5 + 16) )
  {
    v6 = *(_QWORD **)(v5 + 32);
    v7 = &v6[*(unsigned int *)(v5 + 40)];
    if ( v7 == sub_25BD100(v6, (__int64)v7, v15) )
      return v2;
    return 0;
  }
  v8 = *(_QWORD *)(v5 + 8);
  v9 = *(unsigned int *)(v5 + 24);
  if ( !(_DWORD)v9 )
    return v2;
  v10 = (v9 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v11 = (__int64 *)(v8 + 8LL * v10);
  v12 = *v11;
  if ( v3 == *v11 )
  {
LABEL_12:
    if ( v11 == (__int64 *)(v8 + 8 * v9) )
      return v2;
    return 0;
  }
  v13 = 1;
  while ( v12 != -4096 )
  {
    v14 = v13 + 1;
    v10 = (v9 - 1) & (v13 + v10);
    v11 = (__int64 *)(v8 + 8LL * v10);
    v12 = *v11;
    if ( v3 == *v11 )
      goto LABEL_12;
    v13 = v14;
  }
  return v2;
}
