// Function: sub_1B67B10
// Address: 0x1b67b10
//
__int64 __fastcall sub_1B67B10(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+10h] [rbp-30h]

  v3 = sub_16321C0(a2, a1[2], a1[3], 0);
  result = 0;
  if ( v3 )
  {
    sub_1B679A0(a2, v3, (__int64)(a1 + 2), (__int64)(a1 + 6));
    v5 = sub_16321C0(a2, a1[6], a1[7], 0);
    if ( v5 )
    {
      v6 = sub_16498B0(v5);
      sub_164B0D0(v3, v6);
      return 1;
    }
    else
    {
      v7 = a1 + 6;
      v8 = 260;
      sub_164B780(v3, (__int64 *)&v7);
      return 1;
    }
  }
  return result;
}
