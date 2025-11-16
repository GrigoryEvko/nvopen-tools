// Function: sub_14DE650
// Address: 0x14de650
//
_QWORD *__fastcall sub_14DE650(__int64 *a1)
{
  __int64 v2; // r13
  _QWORD *result; // rax
  _QWORD *v4; // rcx
  _QWORD *v5; // rdx
  __int64 v6; // rdi

  v2 = (unsigned int)dword_4F9D900;
  *a1 = 0;
  result = (_QWORD *)sub_2207820(16 * v2);
  v4 = result;
  if ( result && v2 )
  {
    v5 = &result[2 * v2];
    do
    {
      *result = 0;
      result += 2;
      *(result - 1) = 0;
    }
    while ( v5 != result );
  }
  v6 = *a1;
  *a1 = (__int64)v4;
  if ( v6 )
    return (_QWORD *)j_j___libc_free_0_0(v6);
  return result;
}
