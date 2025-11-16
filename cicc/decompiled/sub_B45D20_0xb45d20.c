// Function: sub_B45D20
// Address: 0xb45d20
//
__int64 __fastcall sub_B45D20(__int64 a1, __int64 a2, int a3, char a4, unsigned int a5)
{
  __int64 v5; // r13
  char v6; // al
  unsigned __int16 v8; // si
  unsigned __int16 v9; // ax
  __int64 v10; // rcx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned int *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int16 v20; // ax
  unsigned __int16 v21; // dx
  unsigned __int16 v22; // ax
  unsigned __int16 v23; // dx
  _QWORD v24[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = a2;
  v6 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 60 )
  {
    if ( v6 == 61 )
    {
      v8 = *(_WORD *)(a1 + 2);
      v9 = *(_WORD *)(v5 + 2);
      if ( (v8 & 1) != (v9 & 1) )
        return 0;
      LOWORD(v10) = *(_WORD *)(a1 + 2);
      _BitScanReverse64(&v11, 1LL << (v9 >> 1));
      a5 = v11 ^ 0x3F;
      goto LABEL_8;
    }
    if ( v6 == 62 )
    {
      v9 = *(_WORD *)(a1 + 2);
      v8 = *(_WORD *)(a2 + 2);
      if ( (v9 & 1) != (v8 & 1) )
        return 0;
      LOWORD(v10) = *(_WORD *)(a1 + 2);
      _BitScanReverse64(&v13, 1LL << (v8 >> 1));
      a5 = v13 ^ 0x3F;
LABEL_8:
      _BitScanReverse64((unsigned __int64 *)&v10, 1LL << ((unsigned __int16)v10 >> 1));
      if ( (63 - ((unsigned __int8)v10 ^ 0x3F) == 63 - (_BYTE)a5 || (_BYTE)a3) && ((v8 >> 7) & 7) == ((v9 >> 7) & 7) )
      {
        LOBYTE(a5) = *(_BYTE *)(a1 + 72) == *(_BYTE *)(v5 + 72);
        return a5;
      }
      return 0;
    }
    if ( (unsigned __int8)(v6 - 82) <= 1u )
    {
      LOBYTE(a5) = (*(_WORD *)(a2 + 2) & 0x3F) == (*(_WORD *)(a1 + 2) & 0x3F);
      return a5;
    }
    switch ( v6 )
    {
      case 'U':
        a2 = *(unsigned __int16 *)(a2 + 2);
        if ( (unsigned int)(a2 & 3) - 1 <= 1 != (*(_WORD *)(a1 + 2) & 3u) - 1 <= 1 )
          return 0;
        LOWORD(a2) = ((unsigned __int16)a2 >> 2) & 0x3FF;
        if ( (_WORD)a2 != ((*(_WORD *)(a1 + 2) >> 2) & 0x3FF) )
          return 0;
        goto LABEL_21;
      case '"':
        if ( ((*(_WORD *)(a2 + 2) >> 2) & 0x3FF) != ((*(_WORD *)(a1 + 2) >> 2) & 0x3FF) )
          return 0;
        v14 = *(_QWORD *)(a1 + 72);
        v15 = *(_QWORD *)(a2 + 72);
        if ( !a4 )
        {
          if ( v14 != v15 )
            return 0;
          return sub_B43AF0(a1, v5);
        }
LABEL_36:
        v24[0] = v14;
        v17 = (unsigned int *)sub_BD5C60(a1, a2);
        v18 = sub_A7AD50(v24, v17, v15);
        v24[2] = v19;
        v24[1] = v18;
        if ( !(_BYTE)v19 )
          return 0;
        return sub_B43AF0(a1, v5);
      case '(':
        if ( ((*(_WORD *)(a2 + 2) >> 2) & 0x3FF) != ((*(_WORD *)(a1 + 2) >> 2) & 0x3FF) )
          return 0;
LABEL_21:
        v14 = *(_QWORD *)(a1 + 72);
        v15 = *(_QWORD *)(v5 + 72);
        if ( !a4 )
        {
          if ( v15 != v14 )
            return 0;
          return sub_B43AF0(a1, v5);
        }
        goto LABEL_36;
    }
    if ( v6 != 94 && v6 != 93 )
    {
      switch ( v6 )
      {
        case '@':
          if ( (*(_WORD *)(a2 + 2) & 7) != (*(_WORD *)(a1 + 2) & 7) )
            return 0;
LABEL_41:
          LOBYTE(a5) = *(_BYTE *)(a2 + 72) == *(_BYTE *)(a1 + 72);
          return a5;
        case 'A':
          v20 = *(_WORD *)(a1 + 2);
          v21 = *(_WORD *)(a2 + 2);
          if ( (v21 & 1) != (v20 & 1)
            || ((v21 & 2) != 0) != ((v20 & 2) != 0)
            || ((v21 >> 2) & 7) != ((v20 >> 2) & 7)
            || (unsigned __int8)v21 >> 5 != (unsigned __int8)v20 >> 5 )
          {
            return 0;
          }
          goto LABEL_41;
        case 'B':
          v22 = *(_WORD *)(a1 + 2);
          v23 = *(_WORD *)(a2 + 2);
          if ( ((v23 >> 4) & 0x1F) != ((v22 >> 4) & 0x1F)
            || (v23 & 1) != (v22 & 1)
            || ((v23 >> 1) & 7) != ((v22 >> 1) & 7) )
          {
            return 0;
          }
          goto LABEL_41;
      }
      if ( v6 != 92 )
      {
        a5 = 1;
        if ( v6 == 63 )
          LOBYTE(a5) = *(_QWORD *)(a2 + 72) == *(_QWORD *)(a1 + 72);
        return a5;
      }
    }
    v16 = *(unsigned int *)(a1 + 80);
    a5 = 0;
    if ( v16 == *(_DWORD *)(a2 + 80) )
    {
      a5 = 1;
      if ( 4 * v16 )
        LOBYTE(a5) = memcmp(*(const void **)(a1 + 72), *(const void **)(a2 + 72), 4 * v16) == 0;
    }
    return a5;
  }
  if ( *(_QWORD *)(a1 + 72) != *(_QWORD *)(a2 + 72) )
    return 0;
  _BitScanReverse64((unsigned __int64 *)&a2, 1LL << *(_WORD *)(a2 + 2));
  _BitScanReverse64(&v12, 1LL << *(_WORD *)(a1 + 2));
  LOBYTE(a5) = 63 - ((unsigned __int8)v12 ^ 0x3F) == 63 - ((unsigned __int8)a2 ^ 0x3F);
  return a3 | a5;
}
