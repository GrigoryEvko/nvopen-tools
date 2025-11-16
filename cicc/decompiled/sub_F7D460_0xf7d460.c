// Function: sub_F7D460
// Address: 0xf7d460
//
__int64 __fastcall sub_F7D460(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int8 v5; // al
  int v6; // edi
  __int64 v7; // r8
  unsigned int v8; // eax
  __int64 v9; // r9
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rsi
  bool v13; // al
  int v14; // r8d
  int v15; // r8d
  __int64 v16; // r10
  unsigned int v17; // edi
  __int64 v18; // r9
  char v19; // si
  int v21; // r11d
  int v22; // r10d
  _QWORD v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+30h] [rbp-60h]
  __int64 v26; // [rsp+40h] [rbp-50h] BYREF
  __int64 v27; // [rsp+48h] [rbp-48h]
  __int64 v28; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 32);
  if ( *(_BYTE *)a2 == 34 )
    v4 = *(_QWORD *)(*(_QWORD *)(a2 - 96) + 56LL);
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    v5 = *(_BYTE *)(v4 - 24);
    if ( v5 != 84 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
  }
  if ( (unsigned int)v5 - 80 <= 1 || v5 == 95 )
  {
    v4 = *(_QWORD *)(v4 + 8);
  }
  else if ( v5 == 39 )
  {
    v4 = sub_AA5190(*(_QWORD *)(a3 + 40));
  }
  while ( 1 )
  {
    if ( v4 )
    {
      v10 = v4 - 24;
      v26 = 0;
      v27 = 0;
      v28 = v4 - 24;
      if ( v4 != -4072 && v4 != -8168 )
        sub_BD73F0((__int64)&v26);
    }
    else
    {
      v26 = 0;
      v10 = 0;
      v27 = 0;
      v28 = 0;
    }
    v11 = *(_DWORD *)(a1 + 88);
    if ( !v11 )
      break;
    v6 = v11 - 1;
    v7 = *(_QWORD *)(a1 + 72);
    v8 = (v11 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v9 = *(_QWORD *)(v7 + 24LL * v8 + 16);
    if ( v28 != v9 )
    {
      v22 = 1;
      while ( v9 != -4096 )
      {
        v8 = v6 & (v22 + v8);
        v9 = *(_QWORD *)(v7 + 24LL * v8 + 16);
        if ( v28 == v9 )
          goto LABEL_13;
        ++v22;
      }
      break;
    }
LABEL_13:
    if ( v28 && v28 != -4096 && v28 != -8192 )
      sub_BD60C0(&v26);
LABEL_17:
    if ( a3 == v10 )
      return v4;
    v4 = *(_QWORD *)(v4 + 8);
  }
  v24[0] = 0;
  v24[1] = 0;
  v25 = v10;
  v12 = v10;
  v13 = v10 != -8192 && v10 != -4096 && v10 != 0;
  if ( v13 )
  {
    sub_BD73F0((__int64)v24);
    v12 = v25;
    v13 = v25 != -4096 && v25 != 0 && v25 != -8192;
  }
  v14 = *(_DWORD *)(a1 + 120);
  if ( v14 )
  {
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a1 + 104);
    v17 = v15 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v18 = *(_QWORD *)(v16 + 24LL * v17 + 16);
    if ( v18 == v12 )
    {
LABEL_28:
      v19 = 1;
      goto LABEL_29;
    }
    v21 = 1;
    while ( v18 != -4096 )
    {
      v17 = v15 & (v21 + v17);
      v18 = *(_QWORD *)(v16 + 24LL * v17 + 16);
      if ( v18 == v12 )
        goto LABEL_28;
      ++v21;
    }
  }
  v19 = 0;
LABEL_29:
  if ( v13 )
    sub_BD60C0(v24);
  if ( v28 && v28 != -8192 && v28 != -4096 )
    sub_BD60C0(&v26);
  if ( v19 )
    goto LABEL_17;
  return v4;
}
