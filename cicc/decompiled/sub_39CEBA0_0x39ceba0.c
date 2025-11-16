// Function: sub_39CEBA0
// Address: 0x39ceba0
//
__int64 __fastcall sub_39CEBA0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rcx
  unsigned __int64 *v6; // r8
  unsigned __int64 v7; // r9
  __int64 v8; // r14

  result = sub_39CB090(a1, a2);
  if ( a3 )
  {
    v8 = result;
    result = sub_39CEB00((__int64)a1, a3, result, v5, v6, v7);
    if ( result )
      return sub_39A3B20((__int64)a1, v8, 100, result);
  }
  return result;
}
