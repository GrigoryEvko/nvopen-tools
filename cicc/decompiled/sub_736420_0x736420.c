// Function: sub_736420
// Address: 0x736420
//
__int64 __fastcall sub_736420(__int64 a1, int a2)
{
  __int64 result; // rax
  char v3; // dl
  __int64 v4; // rsi
  char v5; // dl
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_8D97B0(a1);
  if ( (_DWORD)result )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x20) == 0 )
  {
    v3 = *(_BYTE *)(a1 + 140);
    if ( (unsigned __int8)(v3 - 9) > 2u && (v3 != 2 || (*(_BYTE *)(a1 + 161) & 8) == 0) )
      return 1;
    if ( a2 == -1 )
    {
      v4 = 0;
      if ( dword_4F077C4 != 2 )
      {
LABEL_8:
        result = 1;
        if ( *(char *)(a1 + 141) < 0 )
          return *(_BYTE *)(v4 + 4) == 1;
        return result;
      }
    }
    else
    {
      v4 = qword_4F04C68[0] + 776LL * a2;
      if ( dword_4F077C4 != 2 )
        goto LABEL_8;
      if ( v4 )
      {
        v5 = *(_BYTE *)(v4 + 4);
        if ( (unsigned __int8)(v5 - 8) > 1u )
        {
          if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
          {
            if ( v5 == 6 )
              return *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) == *(_QWORD *)(v4 + 208);
          }
          else if ( (v5 & 0xFD) != 5 )
          {
            result = 1;
            if ( (*(_BYTE *)(a1 + 89) & 1) != 0 && (*(_BYTE *)(v4 + 6) & 2) != 0 )
              return dword_4F07590 != 0;
          }
        }
        return result;
      }
    }
    result = 1;
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
      return *((_DWORD *)sub_735B90(-1, a1, v6) + 60) != -1;
  }
  return result;
}
