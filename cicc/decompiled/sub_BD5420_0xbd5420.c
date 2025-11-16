// Function: sub_BD5420
// Address: 0xbd5420
//
char __fastcall sub_BD5420(unsigned __int8 *a1, __int64 a2)
{
  int v3; // edx
  int v4; // eax
  unsigned __int16 v5; // ax
  __int64 v6; // r12
  unsigned __int8 v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned int v13; // r12d
  unsigned __int64 v14; // rdx
  __int64 v15; // r14
  int v16; // edx
  char v17; // al
  unsigned __int8 *v18; // r14
  __int64 **v19; // rax
  __int64 v20; // rax
  unsigned int v21; // edx
  unsigned int v22; // ecx
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v27; // r12d
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *a1;
  if ( (_BYTE)v3 == 3 )
  {
    v5 = (*((_WORD *)a1 + 17) >> 1) & 0x3F;
    if ( v5 )
      goto LABEL_18;
    v15 = *((_QWORD *)a1 + 3);
    v16 = *(unsigned __int8 *)(v15 + 8);
    if ( (_BYTE)v16 != 12 && (unsigned __int8)v16 > 3u && (_BYTE)v16 != 5 && (v16 & 0xFB) != 0xA && (v16 & 0xFD) != 4 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v15 + 8) - 15) > 3u && v16 != 20 )
        goto LABEL_45;
      if ( !(unsigned __int8)sub_BCEBA0(*((_QWORD *)a1 + 3), 0) )
      {
        LOBYTE(v5) = 0;
        return v5;
      }
    }
    if ( (a1[32] & 0xF) != 1 && !sub_B2FC80((__int64)a1) )
    {
      v17 = a1[32] & 0xF;
      if ( ((v17 + 14) & 0xFu) > 3 && ((v17 + 7) & 0xFu) > 1 )
      {
        LOBYTE(v5) = sub_AE5270(a2, (__int64)a1);
        return v5;
      }
    }
    v8 = v15;
LABEL_43:
    LOBYTE(v5) = sub_AE5020(a2, v8);
    return v5;
  }
  if ( (v3 & 0xFD) == 0 )
  {
    if ( !(_BYTE)v3 )
    {
      if ( *(_BYTE *)(a2 + 19) )
        LOBYTE(v3) = *(_BYTE *)(a2 + 18);
      v4 = *(_DWORD *)(a2 + 20);
      if ( !v4 )
        goto LABEL_9;
      if ( v4 != 1 )
        BUG();
      v5 = (*((_WORD *)a1 + 17) >> 1) & 0x3F;
      if ( !v5 || (LOBYTE(v5) = v5 - 1, (unsigned __int8)v3 >= (unsigned __int8)v5) )
LABEL_9:
        LOBYTE(v5) = v3;
      return v5;
    }
    v5 = (*((_WORD *)a1 + 17) >> 1) & 0x3F;
    if ( !v5 )
    {
LABEL_45:
      LOBYTE(v5) = 0;
      return v5;
    }
LABEL_18:
    LOBYTE(v5) = v5 - 1;
    return v5;
  }
  if ( (_BYTE)v3 == 22 )
  {
    v5 = sub_B2BD00((__int64)a1);
    if ( HIBYTE(v5) )
      return v5;
    if ( !(unsigned __int8)sub_B2D720((__int64)a1) )
      goto LABEL_45;
    v6 = sub_B2BD30((__int64)a1);
    v7 = *(_BYTE *)(v6 + 8);
    if ( v7 != 12 && v7 > 3u && v7 != 5 && (v7 & 0xFB) != 0xA && (v7 & 0xFD) != 4 )
    {
      if ( (unsigned __int8)(v7 - 15) > 3u && v7 != 20 )
        goto LABEL_45;
      if ( !(unsigned __int8)sub_BCEBA0(v6, 0) )
      {
        LOBYTE(v5) = 0;
        return v5;
      }
    }
    v8 = v6;
    goto LABEL_43;
  }
  if ( (_BYTE)v3 == 60 )
  {
    LOWORD(v22) = *((_WORD *)a1 + 1);
    goto LABEL_52;
  }
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (unsigned __int8)v3 > 0x15u )
      goto LABEL_45;
    v18 = sub_BD3990(a1, a2);
    v19 = (__int64 **)sub_AE4450(a2, *((_QWORD *)a1 + 1));
    v20 = sub_AD4C50((unsigned __int64)v18, v19, 1);
    if ( !v20 || *(_BYTE *)v20 != 17 )
      goto LABEL_45;
    v21 = *(_DWORD *)(v20 + 32);
    if ( v21 <= 0x40 )
    {
      _RAX = *(_QWORD *)(v20 + 24);
      v22 = 64;
      __asm { tzcnt   rsi, rax }
      if ( _RAX )
        v22 = _RSI;
      if ( v21 <= v22 )
        v22 = v21;
    }
    else
    {
      v22 = sub_C44590(v20 + 24);
    }
    LOBYTE(v5) = 32;
    if ( v22 > 0x1F )
      return v5;
LABEL_52:
    v14 = 1LL << v22;
    goto LABEL_32;
  }
  if ( (unsigned __int8)(v3 - 34) <= 0x33u )
  {
    v23 = 0x8000000000041LL;
    if ( _bittest64(&v23, (unsigned int)(v3 - 34)) )
    {
      v5 = sub_A74820((_QWORD *)a1 + 9);
      if ( HIBYTE(v5) )
        return v5;
      v24 = *((_QWORD *)a1 - 4);
      if ( !v24 )
        goto LABEL_45;
      if ( !*(_BYTE *)v24 && *(_QWORD *)(v24 + 24) == *((_QWORD *)a1 + 10) )
      {
        v29[0] = *(_QWORD *)(v24 + 120);
        v5 = sub_A74820(v29);
        if ( HIBYTE(v5) )
          return v5;
        v24 = *((_QWORD *)a1 - 4);
        if ( !v24 )
        {
          LOBYTE(v5) = 0;
          return v5;
        }
      }
      if ( *(_BYTE *)v24 || *(_QWORD *)(v24 + 24) != *((_QWORD *)a1 + 10) )
        goto LABEL_45;
      v29[0] = *(_QWORD *)(v24 + 120);
      v5 = sub_A74820(v29);
      if ( !HIBYTE(v5) )
        LOBYTE(v5) = 0;
      return v5;
    }
  }
  if ( (_BYTE)v3 != 61 )
    goto LABEL_45;
  if ( (a1[7] & 0x20) == 0 )
    goto LABEL_45;
  v9 = sub_B91C10((__int64)a1, 17);
  if ( !v9 )
    goto LABEL_45;
  v10 = *(_BYTE *)(v9 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(_QWORD *)(v9 - 32);
  else
    v11 = v9 - 8LL * ((v10 >> 2) & 0xF) - 16;
  v12 = *(_QWORD *)(*(_QWORD *)v11 + 136LL);
  v13 = *(_DWORD *)(v12 + 32);
  if ( v13 > 0x40 )
  {
    v27 = v13 - sub_C444A0(v12 + 24);
    LOBYTE(v5) = 63;
    if ( v27 > 0x40 )
      return v5;
    v14 = **(_QWORD **)(v12 + 24);
  }
  else
  {
    v14 = *(_QWORD *)(v12 + 24);
  }
  LOBYTE(v5) = -1;
  if ( v14 )
  {
LABEL_32:
    _BitScanReverse64(&v14, v14);
    LOBYTE(v5) = 63 - (v14 ^ 0x3F);
  }
  return v5;
}
