// Function: sub_85B450
// Address: 0x85b450
//
__int64 __fastcall sub_85B450(__int64 a1, int a2, int a3)
{
  _BYTE *v4; // r13
  int v5; // r14d
  __int64 result; // rax
  char v7; // dl
  __int64 v8; // rax
  int v9; // [rsp+8h] [rbp-38h]
  __int16 v10; // [rsp+Ch] [rbp-34h]

  v4 = *(_BYTE **)(a1 + 32);
  v5 = unk_4D03FE8;
  if ( !unk_4D03FE8 )
  {
LABEL_30:
    if ( !a2 )
      goto LABEL_6;
LABEL_31:
    sub_7345A0((_QWORD *)a1);
    if ( !a3 )
      goto LABEL_7;
    goto LABEL_32;
  }
  if ( a2 )
  {
    v5 = sub_7E16F0();
    if ( !v5 )
      goto LABEL_31;
    v5 = 0;
    goto LABEL_29;
  }
  v5 = 0;
  if ( (v4[195] & 8) == 0
    && dword_4F04C44 == -1
    && (a3 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0) )
  {
    v9 = dword_4F07508[0];
    v10 = dword_4F07508[1];
    sub_7F4BA0(*(_QWORD *)(a1 + 32), (_QWORD *)a1);
    dword_4F07508[0] = v9;
    LOWORD(dword_4F07508[1]) = v10;
    v5 = 1;
  }
  if ( sub_7E16F0() )
  {
LABEL_29:
    sub_7E9A00(a1);
    goto LABEL_30;
  }
LABEL_6:
  sub_72A130(a1);
  if ( !a3 )
  {
LABEL_7:
    sub_72AC90();
    *(_BYTE *)(a1 + 29) |= 1u;
    sub_766370(a1);
    if ( (v4[193] & 0x20) != 0 && v5 )
      sub_760730((__int64)v4);
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(result + 6) & 2) != 0 )
      return result;
    goto LABEL_11;
  }
LABEL_32:
  *(_BYTE *)(a1 + 29) |= 1u;
  sub_766370(a1);
  if ( (v4[193] & 0x20) != 0 && v5 )
    sub_760730((__int64)v4);
LABEL_11:
  result = (unsigned __int8)v4[195];
  if ( (result & 8) != 0 )
    return result;
  if ( (v4[88] & 8) != 0 )
    return sub_7604D0((__int64)v4, 0xBu);
  if ( (v4[198] & 0x20) != 0 )
  {
    sub_7605A0((__int64)v4);
    return sub_7604D0((__int64)v4, 0xBu);
  }
  if ( (*(v4 - 8) & 0x10) != 0 || (v4[200] & 0x18) != 0 )
    return sub_7604D0((__int64)v4, 0xBu);
  v7 = v4[172];
  if ( v7 == 2 )
  {
    if ( (*((_WORD *)v4 + 100) & 0x220) == 0 )
      return result;
    return sub_7604D0((__int64)v4, 0xBu);
  }
  if ( !v7 && (v4[194] & 2) == 0 )
  {
    result &= 3u;
    if ( (char)v4[192] >= 0 )
      goto LABEL_21;
    if ( (_BYTE)result == 1 && dword_4D04824 )
    {
      if ( !unk_4D03FE8 )
        return sub_7604D0((__int64)v4, 0xBu);
      goto LABEL_22;
    }
    if ( dword_4F077C0 && (v4[203] & 0x20) == 0 || dword_4F077C4 != 2 && unk_4F07778 > 199900 && (v4[203] & 0x20) == 0 )
    {
LABEL_21:
      if ( !unk_4D03FE8 || (_BYTE)result != 1 )
        return sub_7604D0((__int64)v4, 0xBu);
LABEL_22:
      v8 = *(_QWORD *)(*(_QWORD *)v4 + 96LL);
      if ( (*(_BYTE *)(v8 + 80) & 8) == 0 && unk_4D04734 != 1 )
      {
        result = *(_QWORD *)(v8 + 16);
        if ( !result )
          return result;
        result = *(_BYTE *)(result + 28) & 6;
        if ( (_BYTE)result != 2 )
          return result;
      }
      return sub_7604D0((__int64)v4, 0xBu);
    }
  }
  return result;
}
