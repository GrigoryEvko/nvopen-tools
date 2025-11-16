// Function: sub_1ADACC0
// Address: 0x1adacc0
//
__int64 __fastcall sub_1ADACC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_1CCABF0(3, a1);
  sub_1CCABF0(3, a2);
  if ( (unsigned __int8)sub_1CCAAC0(1, a2) )
    sub_1CCAB50(1, a1);
  if ( (unsigned __int8)sub_1CCAAC0(1, a1)
    && !(unsigned __int8)sub_1CCAAC0(2, a1)
    && (unsigned __int8)sub_1CCAAC0(1, a1) )
  {
    sub_1ADA5D0(a1);
  }
  result = sub_1CCAAC0(1, a2);
  if ( (_BYTE)result )
    return sub_1ADA5D0(a2);
  return result;
}
