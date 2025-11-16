// Function: sub_C93130
// Address: 0xc93130
//
__int64 *__fastcall sub_C93130(__int64 *a1, __int64 a2)
{
  char *v2; // r9
  __int64 v3; // rcx

  v2 = *(char **)a2;
  v3 = *(_QWORD *)(a2 + 8);
  *a1 = (__int64)(a1 + 2);
  sub_C92B90(a1, v2, (__int64 (__fastcall *)(__int64))sub_C92B70, &v2[v3]);
  return a1;
}
