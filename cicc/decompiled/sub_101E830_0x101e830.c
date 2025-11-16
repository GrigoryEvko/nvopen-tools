// Function: sub_101E830
// Address: 0x101e830
//
unsigned __int8 *__fastcall sub_101E830(unsigned int a1, _BYTE *a2, _BYTE *a3, char a4, __m128i *a5)
{
  unsigned __int8 *result; // rax
  _BYTE *v8; // [rsp+10h] [rbp-20h] BYREF
  _BYTE *v9; // [rsp+18h] [rbp-18h] BYREF

  if ( a1 == 18 )
  {
    v9 = a3;
    v8 = a2;
    result = (unsigned __int8 *)sub_FFE3E0(0x12u, &v8, &v9, a5->m128i_i64);
    if ( !result )
      return sub_1009850((__int64)v8, (__int64)v9, a4, a5, 0, 1);
  }
  else
  {
    if ( a1 > 0x12 )
    {
      if ( a1 == 21 )
        return sub_1009F30(a2, a3, a4, a5->m128i_i64, 0, 1);
      return sub_101AFF0(a1, (__int64 *)a2, (__int64 *)a3, a5, 3u);
    }
    if ( a1 != 14 )
    {
      if ( a1 == 16 )
        return (unsigned __int8 *)sub_10088F0((__int64 *)a2, (__int64 *)a3, a4, a5, 0, 1);
      return sub_101AFF0(a1, (__int64 *)a2, (__int64 *)a3, a5, 3u);
    }
    return sub_100E540((__int64 *)a2, a3, a4, a5, 0, 1);
  }
  return result;
}
