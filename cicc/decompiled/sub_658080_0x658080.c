// Function: sub_658080
// Address: 0x658080
//
__int64 __fastcall sub_658080(_BYTE *a1, int a2)
{
  __int64 result; // rax

  if ( (a1[172] & 0x20) != 0 )
  {
    result = (__int64)&qword_4D045BC + 4;
    if ( !HIDWORD(qword_4D045BC) )
    {
      result = a1[156] & 3;
      if ( (_BYTE)result == 1 )
      {
        result = ((__int64 (*)(void))sub_736990)();
        if ( !(_DWORD)result )
        {
          result = a1[88] & 0x70;
          if ( (_BYTE)result != 16 )
            return sub_684AA0(7, 3648, a1 + 64);
        }
      }
    }
  }
  else if ( !a2 && (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 2) != 0 )
  {
    return sub_686A30(8, 1655, dword_4F07508, *(_QWORD *)a1 + 48LL);
  }
  else
  {
    if ( !HIDWORD(qword_4D045BC) && (a1[156] & 3) == 1 && !(unsigned int)sub_736990(a1) && (a1[88] & 0x70) != 0x10 )
      sub_684AA0(7, 3648, a1 + 64);
    a1[172] |= 0x20u;
    result = dword_4F068E8;
    if ( dword_4F068E8 )
    {
      if ( (a1[170] & 0x20) == 0 )
        return sub_89A0D0(a1);
    }
  }
  return result;
}
