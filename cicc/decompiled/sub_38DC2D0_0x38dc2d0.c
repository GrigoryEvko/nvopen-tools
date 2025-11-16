// Function: sub_38DC2D0
// Address: 0x38dc2d0
//
unsigned __int64 __fastcall sub_38DC2D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 result; // rax
  void (*v7)(); // rdx

  v3 = a1[1];
  v4 = sub_38CF310(a3, 0, v3, 0);
  v5 = sub_38CF310(a2, 0, a1[1], 0);
  result = sub_38CB1F0(17, v5, v4, v3, 0);
  v7 = *(void (**)())(*a1 + 432);
  if ( v7 != nullsub_1937 )
    return ((__int64 (__fastcall *)(__int64 *, unsigned __int64))v7)(a1, result);
  return result;
}
