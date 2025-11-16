// Function: sub_2E32080
// Address: 0x2e32080
//
__int64 *__fastcall sub_2E32080(__int64 *a1, __int64 *a2)
{
  __int64 *result; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // rax

  result = (__int64 *)a1[1];
  if ( a2 != result && a1 != a2 && a2 != result && a1 != result )
  {
    v3 = *result & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 8) = result;
    *result = *result & 7 | *a1 & 0xFFFFFFFFFFFFFFF8LL;
    v4 = *a2;
    *(_QWORD *)(v3 + 8) = a2;
    v4 &= 0xFFFFFFFFFFFFFFF8LL;
    *a1 = v4 | *a1 & 7;
    *(_QWORD *)(v4 + 8) = a1;
    result = (__int64 *)(v3 | *a2 & 7);
    *a2 = (__int64)result;
  }
  return result;
}
