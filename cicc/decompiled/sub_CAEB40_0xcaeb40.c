// Function: sub_CAEB40
// Address: 0xcaeb40
//
__int64 __fastcall sub_CAEB40(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8

  result = sub_CAE820(a1, a2, a3, a4, a5);
  if ( result )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
    result = (__int64)sub_CAE940(a1, a2, v6, v7, v8);
    if ( result )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
  }
  return result;
}
