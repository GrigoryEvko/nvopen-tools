// Function: sub_EAD930
// Address: 0xead930
//
char __fastcall sub_EAD930(__int64 a1, __int64 *a2, _QWORD *a3)
{
  char result; // al

  *a2 = 0;
  result = sub_EAD8C0(a1, a2, a3);
  if ( !result )
    return sub_EAC330(a1, 1u, a2, (__int64)a3);
  return result;
}
