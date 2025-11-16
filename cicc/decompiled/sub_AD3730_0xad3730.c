// Function: sub_AD3730
// Address: 0xad3730
//
__int64 __fastcall sub_AD3730(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD **v3; // rax

  result = sub_ACE990(a1, a2);
  if ( !result )
  {
    v3 = (_QWORD **)sub_BCDA70(*(_QWORD *)(*a1 + 8), (unsigned int)a2);
    return sub_AD3580(**v3 + 1808LL, (__int64)v3, a1, a2);
  }
  return result;
}
