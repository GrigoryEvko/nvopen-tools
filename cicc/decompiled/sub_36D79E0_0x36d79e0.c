// Function: sub_36D79E0
// Address: 0x36d79e0
//
__int64 __fastcall sub_36D79E0(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // cx
  __int64 result; // rax
  _QWORD v4[2]; // [rsp+0h] [rbp-10h] BYREF

  v4[0] = a1;
  v4[1] = a2;
  if ( !(_WORD)a1 )
    return 2 * (unsigned int)((unsigned __int8)sub_3007030((__int64)v4) != 0);
  v2 = a1 - 10;
  if ( (unsigned __int16)(a1 - 126) > 0x31u && v2 > 6u && (unsigned __int16)(a1 - 208) > 0x14u )
    return 0;
  result = 3;
  if ( (_WORD)a1 != 127 )
  {
    if ( (unsigned __int16)a1 > 0x7Fu )
      return (unsigned int)((_WORD)a1 == 138) + 2;
    else
      return (unsigned int)(v2 < 2u) + 2;
  }
  return result;
}
