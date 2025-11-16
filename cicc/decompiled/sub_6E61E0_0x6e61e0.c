// Function: sub_6E61E0
// Address: 0x6e61e0
//
__int64 __fastcall sub_6E61E0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int *v3; // r14
  unsigned int v5; // eax
  __int64 result; // rax
  unsigned int v7; // [rsp+Ch] [rbp-24h] BYREF

  v3 = &v7;
  v7 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v3 = 0;
  v5 = sub_6E6010();
  sub_876E10(a1, a1, a2, v5, a3, v3);
  result = v7;
  if ( v7 )
    return sub_6E50A0();
  return result;
}
