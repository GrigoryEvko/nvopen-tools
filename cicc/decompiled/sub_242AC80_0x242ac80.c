// Function: sub_242AC80
// Address: 0x242ac80
//
__int64 __fastcall sub_242AC80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  unsigned __int64 v7; // rax
  int v8; // ecx
  _BYTE *v9; // r12
  _BYTE *v10; // rax
  __int64 v11; // rax
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rcx
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rsi
  int v22; // edi
  __int64 v23; // rax
  _QWORD v25[4]; // [rsp+10h] [rbp-90h] BYREF
  int v26; // [rsp+30h] [rbp-70h]
  char v27; // [rsp+34h] [rbp-6Ch]
  void *v28; // [rsp+40h] [rbp-60h] BYREF
  __int16 v29; // [rsp+60h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 40) || *(_BYTE *)(a2 + 41) )
    return 0;
  v4 = *(_QWORD *)a2;
  v5 = *(_QWORD *)(a2 + 8);
  if ( !*(_QWORD *)a2 )
    return v5;
  if ( !v5 )
    return *(_QWORD *)a2;
  v7 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 + 48 == v7 )
  {
    v9 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = 0;
    v10 = (_BYTE *)(v7 - 24);
    if ( (unsigned int)(v8 - 30) < 0xB )
      v9 = v10;
  }
  if ( (unsigned int)sub_B46E30((__int64)v9) > 1 )
    goto LABEL_10;
  v17 = *(_DWORD *)(a3 + 24);
  v18 = *(_QWORD *)(a3 + 8);
  if ( !v17 )
  {
LABEL_26:
    v5 = v4;
    v23 = sub_AA5190(v4);
    if ( v23 && v4 + 48 == v23 )
      return 0;
    return v5;
  }
  v19 = v17 - 1;
  v20 = (v17 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v21 = *(_QWORD *)(v18 + 8LL * v20);
  if ( v4 != v21 )
  {
    v22 = 1;
    while ( v21 != -4096 )
    {
      v20 = v19 & (v22 + v20);
      v21 = *(_QWORD *)(v18 + 8LL * v20);
      if ( v4 == v21 )
        goto LABEL_10;
      ++v22;
    }
    goto LABEL_26;
  }
LABEL_10:
  if ( *(_BYTE *)(a2 + 42) )
  {
    v13 = sub_D0E820(v4, v5);
    if ( *v9 == 33 )
      return 0;
    v27 = 1;
    v29 = 257;
    memset(v25, 0, sizeof(v25));
    v26 = 0;
    v14 = sub_F451F0((__int64)v9, v13, (__int64)v25, &v28);
    v15 = v14;
    if ( !v14 )
      return 0;
    sub_242A560(a1, v4, v14, 0);
    v16 = v5;
    v5 = v15;
    *(_BYTE *)(sub_242A560(a1, v15, v16, 0) + 40) = 1;
    *(_BYTE *)(a2 + 41) = 1;
    v11 = sub_AA5190(v15);
    if ( !v11 )
      return v5;
LABEL_12:
    if ( v11 != v5 + 48 )
      return v5;
    return 0;
  }
  v11 = sub_AA5190(v5);
  if ( v11 )
    goto LABEL_12;
  return v5;
}
