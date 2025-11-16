// Function: sub_12FB9C0
// Address: 0x12fb9c0
//
__int64 __fastcall sub_12FB9C0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // [rsp+0h] [rbp-20h]

  result = a1 & 0x7FFF;
  if ( result == 0x7FFF && (a2 & 0x4000000000000000LL) == 0 )
  {
    result = 0x3FFFFFFFFFFFFFFFLL;
    if ( (a2 & 0x3FFFFFFFFFFFFFFFLL) != 0 )
    {
      v4 = a3;
      result = (__int64)sub_12F9B70(16);
      a3 = v4;
    }
  }
  *(_QWORD *)(a3 + 8) = 0;
  *(_QWORD *)(a3 + 16) = 2 * a2;
  *(_BYTE *)a3 = a1 >> 15 != 0;
  return result;
}
