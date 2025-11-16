// Function: sub_A18930
// Address: 0xa18930
//
__int64 __fastcall sub_A18930(__int64 *a1, __int64 a2, char a3)
{
  unsigned int v4; // ebx
  unsigned __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v13; // r15
  int v14; // ebx
  unsigned int v15; // r14d
  __int64 v16; // rbx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  int v19; // r14d
  bool v20; // cc
  unsigned __int64 v21; // rax

  v4 = *(_DWORD *)(a2 + 8);
  if ( !a3 )
  {
    if ( v4 <= 0x40 )
    {
      v5 = *(_QWORD *)a2;
      goto LABEL_4;
    }
    goto LABEL_12;
  }
  sub_A188E0((__int64)a1, v4);
  v20 = v4 <= 0x40;
  v4 = *(_DWORD *)(a2 + 8);
  if ( v20 )
  {
    v5 = *(_QWORD *)a2;
    if ( v4 > 0x40 )
    {
      v7 = *(_QWORD *)v5;
      if ( *(__int64 *)v5 < 0 )
      {
LABEL_6:
        v6 = -2 * v7 + 1;
        goto LABEL_7;
      }
LABEL_29:
      v6 = 2 * v7;
LABEL_7:
      sub_A188E0((__int64)a1, v6);
      v8 = *(_DWORD *)(a2 + 24);
      v9 = *(__int64 **)(a2 + 16);
      if ( v8 > 0x40 )
      {
        v11 = *v9;
        if ( *v9 < 0 )
          goto LABEL_10;
      }
      else
      {
        v10 = 0;
        if ( !v8 )
          return sub_A188E0((__int64)a1, v10);
        v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v8)) >> (64 - (unsigned __int8)v8);
        if ( v11 < 0 )
        {
LABEL_10:
          v10 = -2 * v11 + 1;
          return sub_A188E0((__int64)a1, v10);
        }
      }
      v10 = 2 * v11;
      return sub_A188E0((__int64)a1, v10);
    }
LABEL_4:
    v6 = 0;
    if ( !v4 )
      goto LABEL_7;
    v7 = (__int64)(v5 << (64 - (unsigned __int8)v4)) >> (64 - (unsigned __int8)v4);
    if ( v7 < 0 )
      goto LABEL_6;
    goto LABEL_29;
  }
  if ( v4 > 0x40 )
  {
LABEL_12:
    v13 = 1;
    v14 = v4 - sub_C444A0(a2);
    if ( !v14 )
      goto LABEL_13;
    goto LABEL_22;
  }
  v13 = 1;
  if ( !*(_QWORD *)a2 )
  {
LABEL_13:
    v15 = *(_DWORD *)(a2 + 24);
    v16 = a2 + 16;
    if ( v15 <= 0x40 )
      goto LABEL_14;
LABEL_23:
    v17 = 0x100000000LL;
    v19 = v15 - sub_C444A0(v16);
    if ( !v19 )
      goto LABEL_17;
    goto LABEL_16;
  }
  _BitScanReverse64(&v21, *(_QWORD *)a2);
  v14 = 64 - (v21 ^ 0x3F);
LABEL_22:
  v15 = *(_DWORD *)(a2 + 24);
  v13 = ((unsigned int)(v14 - 1) >> 6) + 1;
  v16 = a2 + 16;
  if ( v15 > 0x40 )
    goto LABEL_23;
LABEL_14:
  v17 = 0x100000000LL;
  v18 = *(_QWORD *)(a2 + 16);
  if ( v18 )
  {
    _BitScanReverse64(&v18, v18);
    v19 = 64 - (v18 ^ 0x3F);
LABEL_16:
    v17 = (unsigned __int64)(((unsigned int)(v19 - 1) >> 6) + 1) << 32;
  }
LABEL_17:
  sub_A188E0((__int64)a1, v13 | v17);
  sub_A16F20(a1, a2);
  return sub_A16F20(a1, v16);
}
