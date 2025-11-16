// Function: sub_651B00
// Address: 0x651b00
//
__int64 __fastcall sub_651B00(unsigned int a1)
{
  unsigned __int64 v1; // rdx
  unsigned int v2; // r12d
  __int64 v4; // rcx
  __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int16 v8; // ax
  _BYTE v9[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( (unsigned __int16)(word_4F06418[0] - 88) <= 0xFu )
  {
    v4 = 36993;
    if ( _bittest64(&v4, (unsigned int)word_4F06418[0] - 88) )
      return 1;
  }
  else if ( unk_4F07758 && word_4F06418[0] == 77 )
  {
    return 1;
  }
  LOBYTE(v1) = 0;
  if ( (unsigned __int16)(word_4F06418[0] - 160) <= 0x22u )
    v1 = (0x701004201uLL >> (LOBYTE(word_4F06418[0]) + 96)) & 1;
  if ( word_4F06418[0] == 244 )
    return 1;
  if ( (_BYTE)v1 )
    return 1;
  v2 = sub_651570(a1 & 1, 0, a1);
  if ( v2 )
    return 1;
  switch ( word_4F06418[0] )
  {
    case 0x19u:
      if ( dword_4D043F8 )
        return (unsigned __int16)sub_7BE840(0, 0) == 25;
      break;
    case 0x8Eu:
    case 0xF8u:
    case 0xBAu:
      return 1;
    case 0xBBu:
      sub_7ADF70(v9, 0);
      sub_7AE360(v9);
      sub_7B8B50(v9, 0, v6, v7);
      v2 = sub_651B00(a1);
      sub_7BC000(v9);
      return v2;
    default:
      if ( word_4F06418[0] == 1
        && ((a1 & 1) == 0 || !dword_4D04428 || (unsigned __int16)sub_7BE840(0, 0) != 73)
        && (unk_4D04A11 & 0x20) == 0
        && (a1 & 2) != 0
        && !*(_QWORD *)(qword_4D04A00 + 24) )
      {
        if ( (a1 & 1) != 0 )
        {
          if ( !*(_QWORD *)(qword_4D04A00 + 32) || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 5) & 4) == 0 )
          {
            v5 = sub_7BE840(0, 0);
            return (v5 == 156) | (unsigned __int8)(v5 == 1);
          }
          return v2;
        }
        v8 = sub_7BE840(0, 0);
        if ( v8 != 1 && v8 != 156 && (unsigned __int16)(v8 - 33) > 1u && (!unk_4D04474 || v8 != 52) )
          return v2;
        return 1;
      }
      break;
  }
  return v2;
}
