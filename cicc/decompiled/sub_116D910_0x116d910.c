// Function: sub_116D910
// Address: 0x116d910
//
_QWORD *__fastcall sub_116D910(_QWORD *a1, __int64 *a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v4; // r9
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v7; // rcx

  result = a1;
  v4 = a2[1];
  v5 = a2[3];
  v6 = *a2;
  v7 = a2[2];
  if ( a3 )
  {
    v7 += 8 * a3;
    v6 += 32 * a3;
  }
  *result = v6;
  result[1] = v7;
  result[2] = v4;
  result[3] = v5;
  return result;
}
