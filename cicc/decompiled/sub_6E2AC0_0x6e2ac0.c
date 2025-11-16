// Function: sub_6E2AC0
// Address: 0x6e2ac0
//
__int64 __fastcall sub_6E2AC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = qword_4D03C50;
  v3 = *(_QWORD *)(qword_4D03C50 + 48LL);
  if ( v3 )
  {
    result = sub_733920(v3);
    if ( !(_DWORD)result )
      return sub_6E2A90(v3, a2);
  }
  return result;
}
