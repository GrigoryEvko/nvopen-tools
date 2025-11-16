// Function: sub_2E79D20
// Address: 0x2e79d20
//
__int64 __fastcall sub_2E79D20(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 64);
  if ( v3 )
    sub_2E79BC0(v3, (__int64)a2);
  sub_2E31100(a2);
  result = *(_QWORD *)(a1 + 312);
  *a2 = result;
  *(_QWORD *)(a1 + 312) = a2;
  return result;
}
