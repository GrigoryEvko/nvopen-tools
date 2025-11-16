// Function: sub_1348600
// Address: 0x1348600
//
__int64 __fastcall sub_1348600(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  sub_134BBB0();
  a1[396] += a2[396];
  a1[397] += a2[397];
  a1[398] += a2[398];
  result = a2[399];
  a1[399] += result;
  return result;
}
