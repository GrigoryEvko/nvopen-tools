// Function: sub_2D19D80
// Address: 0x2d19d80
//
_BOOL8 __fastcall sub_2D19D80(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  bool v5; // r15
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // r14
  _QWORD *v10; // rdx
  int v11; // r10d
  unsigned int v12; // ecx
  unsigned __int64 v13; // rax
  __int16 v14; // dx
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  int v19; // r10d
  char v20; // al
  int v21; // [rsp+0h] [rbp-50h]
  unsigned int v22; // [rsp+8h] [rbp-48h]
  int v23; // [rsp+8h] [rbp-48h]
  bool v24; // [rsp+Fh] [rbp-41h]
  _QWORD v25[8]; // [rsp+10h] [rbp-40h] BYREF

  v1 = *(_QWORD *)(a1 + 40) + 312LL;
  v2 = *(_QWORD *)(a1 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  v24 = 0;
  v4 = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  while ( v4 != v3 )
  {
    if ( !v3 )
      BUG();
    if ( *(_BYTE *)(v3 - 24) != 60 )
      goto LABEL_15;
    v5 = sub_B4D040(v3 - 24);
    if ( !v5 )
      goto LABEL_15;
    v6 = *(_QWORD *)(v3 + 48);
    if ( *(_BYTE *)(v6 + 8) != 16 )
      goto LABEL_15;
    _BitScanReverse64((unsigned __int64 *)&v7, 1LL << *(_WORD *)(v3 - 22));
    v8 = *(_QWORD *)(v3 - 56);
    v9 = 0x8000000000000000LL >> ((unsigned __int8)v7 ^ 0x3Fu);
    v10 = *(_QWORD **)(v8 + 24);
    if ( *(_DWORD *)(v8 + 32) > 0x40u )
      v10 = (_QWORD *)*v10;
    v11 = (int)v10;
    if ( (_BYTE)qword_50161C8 )
    {
      v12 = sub_2D19A70(v1, v9, (int)v10, *(_QWORD *)(v3 + 48));
    }
    else
    {
      LODWORD(v7) = 0x8000000000000000LL >> ((unsigned __int8)v7 ^ 0x3Fu);
      if ( !(_DWORD)v9 )
      {
        v23 = (int)v10;
        v20 = sub_AE5260(v1, v6);
        v11 = v23;
        v7 = 1LL << v20;
      }
      v22 = v7;
      v21 = v11;
      v16 = sub_9208B0(v1, v6);
      v25[1] = v17;
      v25[0] = (unsigned __int64)(v16 + 7) >> 3;
      v18 = sub_CA1930(v25);
      v12 = v22;
      v19 = v18 * v21;
      if ( v22 <= 0xF )
      {
        if ( (v19 & 0xF) == 0 )
        {
          v13 = 16;
LABEL_24:
          if ( (_DWORD)v9 == (_DWORD)v13 )
            goto LABEL_15;
LABEL_13:
          _BitScanReverse64(&v13, v13);
          v14 = 63 - (v13 ^ 0x3F);
LABEL_14:
          v24 = v5;
          *(_WORD *)(v3 - 22) = v14 | *(_WORD *)(v3 - 22) & 0xFFC0;
          goto LABEL_15;
        }
        LODWORD(v13) = 16;
        while ( 1 )
        {
          v13 = (unsigned int)v13 >> 1;
          if ( v22 >= (unsigned int)v13 )
            break;
          if ( (v19 & ((_DWORD)v13 - 1)) == 0 )
            goto LABEL_24;
        }
      }
    }
    if ( (_DWORD)v9 != v12 )
    {
      v13 = v12;
      v14 = 255;
      if ( v12 )
        goto LABEL_13;
      goto LABEL_14;
    }
LABEL_15:
    v3 = *(_QWORD *)(v3 + 8);
  }
  return v24;
}
