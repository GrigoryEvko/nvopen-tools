// Function: sub_7E7CB0
// Address: 0x7e7cb0
//
__int64 __fastcall sub_7E7CB0(__int64 a1)
{
  __int64 result; // rax
  __m128i *v2; // [rsp+8h] [rbp-28h]
  _QWORD *v3; // [rsp+18h] [rbp-18h] BYREF

  v3 = 0;
  result = sub_7E2100(a1, &v3);
  if ( result )
  {
    *((_BYTE *)v3 + 16) = 1;
  }
  else
  {
    v2 = sub_7E7CA0(a1);
    sub_7E2130((__int64)v2);
    return (__int64)v2;
  }
  return result;
}
