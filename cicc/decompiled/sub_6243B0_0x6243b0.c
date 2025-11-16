// Function: sub_6243B0
// Address: 0x6243b0
//
__int64 __fastcall sub_6243B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rdi

  v4 = a1;
  if ( dword_4F077C4 != 2 || (result = sub_8D3A70(a1), !(_DWORD)result) && (result = sub_8D3D40(a1), !(_DWORD)result) )
  {
    if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
    {
      result = sub_8D4C10(a1, dword_4F077C4 != 2);
      if ( (_DWORD)result == 4 )
        goto LABEL_14;
      result = sub_8D32E0(a1);
      if ( !a2 )
        goto LABEL_14;
    }
    else
    {
      result = sub_8D32E0(a1);
      if ( !a2 )
        goto LABEL_14;
    }
    if ( !(_DWORD)result && (*(_BYTE *)(a2 + 120) & 0x7F) != 0 )
    {
      if ( dword_4F077C4 != 2 )
      {
        for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
          ;
        if ( (unsigned int)sub_8D2600(a1)
          && (*(_BYTE *)(v4 + 140) & 0xFB) == 8
          && (unsigned int)sub_8D4C10(v4, dword_4F077C4 != 2) == 2 )
        {
          v6 = 4;
LABEL_32:
          result = sub_684AA0(v6, 815, a2 + 72);
          goto LABEL_14;
        }
      }
      if ( unk_4F04C48 == -1
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0
        || (result = 776LL * (int)dword_4F04C5C, (*(_BYTE *)(qword_4F04C68[0] + result + 6) & 2) != 0) )
      {
        v6 = 5;
        goto LABEL_32;
      }
    }
  }
LABEL_14:
  if ( dword_4F077C4 == 2 )
  {
    result = *(_BYTE *)(v4 + 140) & 0xFB;
    if ( (*(_BYTE *)(v4 + 140) & 0xFB) == 8 )
    {
      result = sub_8D4C10(v4, 0);
      if ( (result & 2) != 0 )
      {
        if ( unk_4F04C48 == -1
          || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0
          || (result = 776LL * (int)dword_4F04C5C, (*(_BYTE *)(qword_4F04C68[0] + result + 6) & 2) != 0) )
        {
          v7 = 4;
          if ( dword_4F077C4 == 2 )
            v7 = (unsigned int)(unk_4F07778 > 202001) + 4;
          return sub_684AA0(v7, 3014, a3);
        }
      }
    }
  }
  return result;
}
