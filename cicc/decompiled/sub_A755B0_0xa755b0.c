// Function: sub_A755B0
// Address: 0xa755b0
//
__int64 __fastcall sub_A755B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_A72650(a1, a2);
  sub_A72590(a1, a2);
  sub_A72890(a1, a2);
  sub_A72710(a1, a2);
  sub_A72420(a1, a2);
  sub_A727D0(a1, a2);
  if ( !(unsigned __int8)sub_B2D610(a1, 30) && (unsigned __int8)sub_B2D610(a2, 30) )
    sub_B2CD30(a1, 30);
  sub_A724E0(a1, a2);
  sub_A72950(a1, a2);
  if ( !(unsigned __int8)sub_B2D610(a1, 68) && (unsigned __int8)sub_B2D610(a2, 68) )
    sub_B2CD30(a1, 68);
  sub_A6E4A0(a1, a2);
  sub_A6DEB0(a1, a2);
  sub_A72260(a1, a2);
  sub_A72330(a1, a2);
  if ( (unsigned __int8)sub_B2F060(a2) && !(unsigned __int8)sub_B2F060(a1) )
    sub_B2CD30(a1, 44);
  result = sub_B2D610(a1, 19);
  if ( (_BYTE)result )
  {
    result = sub_B2D610(a2, 19);
    if ( !(_BYTE)result )
      return sub_B2D470(a1, 19);
  }
  return result;
}
