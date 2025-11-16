// Function: sub_1649510
// Address: 0x1649510
//
__int64 __fastcall sub_1649510(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 result; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // ecx
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // r12d
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r12
  __int16 v18; // cx
  __int64 v19; // r12
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned int v23; // r12d
  _QWORD v24[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_BYTE *)(a1 + 16);
  if ( v3 != 3 )
  {
    switch ( v3 )
    {
      case 0u:
        return 0;
      case 0x11u:
        result = sub_15E0370(a1);
        if ( !(_DWORD)result )
        {
          if ( !(unsigned __int8)sub_15E04F0(a1) )
            return 0;
          v5 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
          v6 = *(unsigned __int8 *)(v5 + 8);
          if ( (unsigned __int8)v6 > 0xFu || (v21 = 35454, !_bittest64(&v21, v6)) )
          {
            if ( (unsigned int)(v6 - 13) > 1 && (_DWORD)v6 != 16 || !sub_16435F0(v5, 0) )
              return 0;
          }
          v7 = v5;
          return sub_15A9FE0(a2, v7);
        }
        return result;
      case 0x35u:
        v18 = *(_WORD *)(a1 + 18);
        result = (unsigned int)(1 << v18) >> 1;
        if ( !((unsigned int)(1 << v18) >> 1) )
        {
          v19 = *(_QWORD *)(a1 + 56);
          v20 = *(unsigned __int8 *)(v19 + 8);
          if ( (unsigned __int8)v20 <= 0xFu )
          {
            v22 = 35454;
            if ( _bittest64(&v22, v20) )
              return sub_15AAE50(a2, v19);
          }
          if ( ((unsigned int)(v20 - 13) <= 1 || (_DWORD)v20 == 16) && sub_16435F0(*(_QWORD *)(a1 + 56), 0) )
            return sub_15AAE50(a2, v19);
          return 0;
        }
        return result;
    }
    if ( v3 > 0x17u )
    {
      if ( v3 == 78 )
      {
        v16 = a1 | 4;
      }
      else
      {
        if ( v3 != 29 )
          goto LABEL_26;
        v16 = a1 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17 )
      {
        v24[0] = *(_QWORD *)(v17 + 56);
        return sub_1560380(v24);
      }
      return 0;
    }
LABEL_26:
    if ( v3 != 54 || !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
      return 0;
    v13 = sub_1625790(a1, 17);
    if ( !v13 )
      return 0;
    v14 = *(_QWORD *)(*(_QWORD *)(v13 - 8LL * *(unsigned int *)(v13 + 8)) + 136LL);
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
      return *(unsigned int *)(v14 + 24);
    v23 = v15 - sub_16A57B0(v14 + 24);
    result = 0xFFFFFFFFLL;
    if ( v23 <= 0x40 )
      return **(unsigned int **)(v14 + 24);
    return result;
  }
  v8 = *(_DWORD *)(a1 + 32) >> 15;
  result = (unsigned int)(1 << v8) >> 1;
  if ( !((unsigned int)(1 << v8) >> 1) )
  {
    v9 = *(_QWORD *)(a1 + 24);
    v10 = *(unsigned __int8 *)(v9 + 8);
    if ( (unsigned __int8)v10 > 0xFu || (v11 = 35454, !_bittest64(&v11, v10)) )
    {
      if ( (unsigned int)(v10 - 13) > 1 && (_DWORD)v10 != 16 || !sub_16435F0(*(_QWORD *)(a1 + 24), 0) )
        return 0;
    }
    if ( (*(_BYTE *)(a1 + 32) & 0xF) != 1 && !sub_15E4F60(a1) )
    {
      v12 = *(_BYTE *)(a1 + 32) & 0xF;
      if ( ((v12 + 14) & 0xFu) > 3 && ((v12 + 7) & 0xFu) > 1 )
        return sub_15AAE60(a2, a1);
    }
    v7 = v9;
    return sub_15A9FE0(a2, v7);
  }
  return result;
}
