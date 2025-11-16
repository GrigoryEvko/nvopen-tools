// Function: sub_B976B0
// Address: 0xb976b0
//
__int64 __fastcall sub_B976B0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v5; // rdi
  __int64 result; // rax

  v5 = sub_B911C0(a2);
  result = 0;
  if ( v5 )
  {
    sub_B97400((__int64)v5, a1, a3);
    return 1;
  }
  return result;
}
