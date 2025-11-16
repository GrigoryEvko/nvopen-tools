// Function: sub_3704C60
// Address: 0x3704c60
//
__int64 *__fastcall sub_3704C60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  sub_370D250(&v5, a2 + 16, a3);
  v3 = v5 | 1;
  if ( (v5 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v3 = 1;
  *a1 = v3;
  return a1;
}
