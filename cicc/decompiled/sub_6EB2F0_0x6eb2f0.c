// Function: sub_6EB2F0
// Address: 0x6eb2f0
//
__int64 __fastcall sub_6EB2F0(int a1, int a2, int a3, int a4)
{
  __int64 v4; // r8
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-18h]
  int v7; // [rsp+1Ch] [rbp-4h] BYREF

  v7 = 0;
  v4 = qword_4D03C50;
  if ( !qword_4D03C50 )
    return sub_6EB250(a1, a2, a3, a4, v4);
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
  {
    v4 = 0;
    return sub_6EB250(a1, a2, a3, a4, v4);
  }
  result = sub_6EB250(a1, a2, a3, a4, (__int64)&v7);
  if ( v7 )
  {
    v6 = result;
    sub_6E50A0();
    return v6;
  }
  return result;
}
