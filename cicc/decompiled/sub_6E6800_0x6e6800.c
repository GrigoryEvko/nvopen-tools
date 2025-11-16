// Function: sub_6E6800
// Address: 0x6e6800
//
__int64 __fastcall sub_6E6800(__m128i *a1, __int64 *a2)
{
  __int64 *v3; // rdi
  __int64 result; // rax

  v3 = sub_6E1C80(a2);
  result = 0;
  if ( v3 )
  {
    sub_6E6610(v3, a1, 1);
    return 1;
  }
  return result;
}
