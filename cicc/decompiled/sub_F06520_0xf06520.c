// Function: sub_F06520
// Address: 0xf06520
//
__int64 __fastcall sub_F06520(char **a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_B19F20(*((_QWORD *)*a1 + 10), a1[1], a2);
  *a1[2] |= result;
  return result;
}
