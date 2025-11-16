// Function: sub_36CCFF0
// Address: 0x36ccff0
//
_BOOL8 __fastcall sub_36CCFF0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rsi
  _QWORD *v6; // r11
  unsigned __int64 v7; // rax
  int v8; // eax
  int v9; // r9d
  __int16 v10; // di
  bool v11; // r13
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  unsigned int v17; // r11d
  char i; // al
  unsigned __int64 v19; // rax
  char v20; // cl
  unsigned __int8 v21; // al
  int v22; // [rsp+Ch] [rbp-54h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  int v24; // [rsp+18h] [rbp-48h]
  char v25; // [rsp+18h] [rbp-48h]
  unsigned __int8 v26; // [rsp+1Eh] [rbp-42h]
  bool v27; // [rsp+1Fh] [rbp-41h]
  _QWORD v28[8]; // [rsp+20h] [rbp-40h] BYREF

  v1 = *(_QWORD *)(a1 + 80);
  v23 = *(_QWORD *)(a1 + 40) + 312LL;
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 32);
  v3 = v1 + 24;
  v27 = 0;
  if ( v2 != v3 )
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      if ( *(_BYTE *)(v2 - 24) != 60 )
        goto LABEL_11;
      v11 = sub_B4D040(v2 - 24);
      if ( !v11 )
        goto LABEL_11;
      if ( (unsigned __int8)sub_B4CE70(v2 - 24) )
        break;
      v12 = *(_QWORD *)(v2 + 48);
      if ( *(_BYTE *)(v12 + 8) == 16 )
      {
        v5 = *(_QWORD *)(v12 + 24);
        LODWORD(v6) = *(_DWORD *)(v12 + 32);
        goto LABEL_6;
      }
LABEL_11:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return v27;
    }
    v4 = *(_QWORD *)(v2 - 56);
    v5 = *(_QWORD *)(v2 + 48);
    v6 = *(_QWORD **)(v4 + 24);
    if ( *(_DWORD *)(v4 + 32) > 0x40u )
      v6 = (_QWORD *)*v6;
LABEL_6:
    _BitScanReverse64(&v7, 1LL << *(_WORD *)(v2 - 22));
    v8 = v7 ^ 0x3F;
    v9 = 63 - v8;
    if ( (_BYTE)qword_50408C8 )
    {
      v10 = (unsigned __int8)v9;
      if ( v9 <= 3 )
      {
        v25 = 63 - v8;
        v21 = sub_36CCDB0(v23, (int)v6, v5);
        LOBYTE(v9) = v25;
        v10 = v21;
      }
      goto LABEL_9;
    }
    v26 = 63 - v8;
    v22 = 63 - v8;
    v24 = (int)v6;
    v14 = sub_9208B0(v23, v5);
    v28[1] = v15;
    v28[0] = (unsigned __int64)(v14 + 7) >> 3;
    v16 = sub_CA1930(v28);
    LOBYTE(v9) = v22;
    v17 = v16 * v24;
    if ( v22 <= 3 )
    {
      if ( (v17 & 0xF) == 0 )
      {
        v10 = 4;
LABEL_9:
        if ( (_BYTE)v9 != (_BYTE)v10 )
        {
          v27 = v11;
          *(_WORD *)(v2 - 22) = *(_WORD *)(v2 - 22) & 0xFFC0 | v10;
        }
        goto LABEL_11;
      }
      for ( i = 60; ; i = v19 ^ 0x3F )
      {
        v20 = 63 - i;
        v10 = (unsigned __int8)(63 - i);
        if ( v26 >= (unsigned __int8)(63 - i) )
          break;
        if ( (v17 & (unsigned __int64)((1LL << v20) - 1)) == 0 )
          goto LABEL_9;
        _BitScanReverse64(&v19, (unsigned __int64)(1LL << v20) >> 1);
      }
    }
    v10 = (unsigned __int8)v22;
    goto LABEL_9;
  }
  return v27;
}
