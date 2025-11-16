// Function: sub_727DD0
// Address: 0x727dd0
//
__int64 __fastcall sub_727DD0(char a1, _QWORD *a2)
{
  char v2; // al
  __int64 v3; // rsi
  __int64 result; // rax

  if ( a2 )
  {
    v2 = *((_BYTE *)a2 - 8);
    if ( (v2 & 1) != 0 )
    {
      if ( (v2 & 2) != 0 )
      {
        if ( *a2 )
        {
          v3 = sub_72B7A0(a2);
        }
        else if ( unk_4D03FE8 )
        {
          v3 = *qword_4D03FD0;
        }
        else
        {
          v3 = unk_4D03FF0;
        }
        result = sub_727B10(24, v3);
      }
      else
      {
        result = sub_7279A0(24);
      }
    }
    else
    {
      result = (__int64)sub_7246D0(24);
    }
  }
  else
  {
    result = (__int64)sub_7247C0(24);
  }
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_BYTE *)(result + 16) = a1;
  *(_BYTE *)(result + 17) = 0;
  return result;
}
