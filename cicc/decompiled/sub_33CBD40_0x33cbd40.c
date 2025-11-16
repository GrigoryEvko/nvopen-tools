// Function: sub_33CBD40
// Address: 0x33cbd40
//
__int64 __fastcall sub_33CBD40(int a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD v4[3]; // [rsp+0h] [rbp-20h] BYREF

  v4[0] = a2;
  v4[1] = a3;
  if ( (_WORD)a2 )
  {
    if ( (unsigned __int16)(a2 - 2) <= 7u
      || (unsigned __int16)(a2 - 17) <= 0x6Cu
      || (unsigned __int16)(a2 - 176) <= 0x1Fu )
    {
      goto LABEL_5;
    }
  }
  else if ( sub_3007070((__int64)v4) )
  {
LABEL_5:
    result = a1 ^ 7u;
    goto LABEL_6;
  }
  result = a1 ^ 0xFu;
LABEL_6:
  if ( (unsigned int)result > 0x17 )
    return (unsigned int)result & 0xFFFFFFF7;
  return result;
}
