// Function: sub_15E7D40
// Address: 0x15e7d40
//
_QWORD *__fastcall sub_15E7D40(__int64 *a1, __int64 *a2, char a3)
{
  _QWORD *result; // rax
  _QWORD *v5; // [rsp+8h] [rbp-18h]

  result = sub_15E7030(a1, 86, a2);
  if ( a3 )
  {
    v5 = result;
    sub_15F2440(result, 2);
    return v5;
  }
  return result;
}
