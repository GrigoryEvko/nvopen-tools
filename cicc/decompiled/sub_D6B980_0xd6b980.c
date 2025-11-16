// Function: sub_D6B980
// Address: 0xd6b980
//
_QWORD *__fastcall sub_D6B980(__int64 a1, _QWORD *a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx

  result = sub_D6B8E0(a1, (__int64)a2);
  if ( v3 )
    return (_QWORD *)sub_D68100(a1, (__int64)result, v3, a2);
  return result;
}
