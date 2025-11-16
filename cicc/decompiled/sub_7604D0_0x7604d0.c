// Function: sub_7604D0
// Address: 0x7604d0
//
__int64 __fastcall sub_7604D0(__int64 a1, unsigned __int8 a2)
{
  __int64 result; // rax

  result = sub_760370((_QWORD *)a1, a2);
  if ( a2 == 11 )
  {
    if ( *(_BYTE *)(a1 + 172) )
      goto LABEL_13;
    if ( dword_4F077C4 == 2 )
    {
      if ( *(char *)(a1 + 192) < 0 )
      {
        result = dword_4D04824;
        if ( dword_4D04824 )
          return sub_7605A0(a1);
      }
      else
      {
        if ( (*(_BYTE *)(a1 + 195) & 1) == 0 )
          return sub_7605A0(a1);
        if ( !(unsigned int)sub_736960(a1) )
          return sub_7605A0(a1);
        result = (__int64)&dword_4D04824;
        if ( dword_4D04824 )
        {
          if ( *(_BYTE *)(a1 + 172) != 2 )
            return sub_7605A0(a1);
        }
      }
      if ( (*(_BYTE *)(a1 + 204) & 1) == 0 )
      {
LABEL_13:
        if ( (*(_BYTE *)(a1 + 196) & 0x18) == 0 && (*(_BYTE *)(a1 + 91) & 4) == 0 && (*(_WORD *)(a1 + 200) & 0x218) == 0 )
          return result;
      }
    }
    return sub_7605A0(a1);
  }
  return result;
}
