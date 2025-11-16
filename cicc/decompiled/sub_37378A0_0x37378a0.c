// Function: sub_37378A0
// Address: 0x37378a0
//
__int64 __fastcall sub_37378A0(__int64 a1, unsigned __int64 **a2, __int16 a3, __int64 a4)
{
  __int64 v5[2]; // [rsp+0h] [rbp-10h] BYREF

  v5[0] = 3;
  HIWORD(v5[0]) = a3;
  v5[1] = a4;
  return sub_3248F80(a2, (__int64 *)(a1 + 88), v5);
}
