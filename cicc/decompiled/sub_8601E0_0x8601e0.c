// Function: sub_8601E0
// Address: 0x8601e0
//
__int64 *__fastcall sub_8601E0(__int64 a1, __int64 *a2)
{
  __int64 *result; // rax
  __int64 i; // rbx

  result = sub_85C120(2u, *(_DWORD *)(a1 + 24), 0, 0, 0, 0, 0, 0, 0, *(char **)(a1 + 88), (__int64 *)a1, (__int64)a2, 0);
  for ( i = *a2; i; i = *(_QWORD *)(i + 16) )
    result = (__int64 *)sub_883680(i);
  if ( *(_QWORD *)(a1 + 88) )
  {
    result = (__int64 *)qword_4F06BC0;
    *(_QWORD *)(qword_4F06BC0 + 40LL) = *(_QWORD *)(*(_QWORD *)(qword_4F06BC0 + 32LL) + 24LL);
  }
  return result;
}
