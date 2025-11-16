// Function: sub_19EDA30
// Address: 0x19eda30
//
void __fastcall sub_19EDA30(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // cc
  unsigned int v4; // r8d
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rsi
  __int64 v11; // r9
  int v12; // r11d
  __int64 *v13; // r10
  int v14; // ecx
  int v15; // ecx
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v19[10]; // [rsp+10h] [rbp-50h] BYREF

  v3 = *(_BYTE *)(a2 + 16) <= 0x17u;
  v18 = a2;
  if ( v3 )
    return;
  v4 = *(_DWORD *)(a1 + 1760);
  v6 = a1 + 1736;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 1736);
LABEL_15:
    sub_19ED880(v6, 2 * v4);
LABEL_16:
    sub_19EAFC0(v6, &v18, v19);
    v10 = (__int64 *)v19[0];
    v7 = v18;
    v15 = *(_DWORD *)(a1 + 1752) + 1;
    goto LABEL_11;
  }
  v7 = a2;
  v8 = *(_QWORD *)(a1 + 1744);
  v9 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v8 + ((unsigned __int64)v9 << 6));
  v11 = *v10;
  if ( v7 == *v10 )
  {
LABEL_4:
    sub_19E5640((__int64)v19, (__int64)(v10 + 1), a3);
    return;
  }
  v12 = 1;
  v13 = 0;
  while ( v11 != -8 )
  {
    if ( !v13 && v11 == -16 )
      v13 = v10;
    v9 = (v4 - 1) & (v12 + v9);
    v10 = (__int64 *)(v8 + ((unsigned __int64)v9 << 6));
    v11 = *v10;
    if ( v7 == *v10 )
      goto LABEL_4;
    ++v12;
  }
  v14 = *(_DWORD *)(a1 + 1752);
  if ( v13 )
    v10 = v13;
  ++*(_QWORD *)(a1 + 1736);
  v15 = v14 + 1;
  if ( 4 * v15 >= 3 * v4 )
    goto LABEL_15;
  if ( v4 - *(_DWORD *)(a1 + 1756) - v15 <= v4 >> 3 )
  {
    sub_19ED880(v6, v4);
    goto LABEL_16;
  }
LABEL_11:
  *(_DWORD *)(a1 + 1752) = v15;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 1756);
  *v10 = v7;
  v16 = v10 + 6;
  v10[1] = 0;
  v17 = (__int64)(v10 + 1);
  *(_QWORD *)(v17 + 8) = v16;
  *(_QWORD *)(v17 + 16) = v16;
  *(_QWORD *)(v17 + 24) = 2;
  *(_DWORD *)(v17 + 32) = 0;
  sub_19E5640((__int64)v19, v17, a3);
}
