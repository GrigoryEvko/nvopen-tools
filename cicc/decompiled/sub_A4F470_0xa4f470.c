// Function: sub_A4F470
// Address: 0xa4f470
//
__int64 __fastcall sub_A4F470(__int64 a1)
{
  _QWORD *v1; // rbx

  v1 = *(_QWORD **)(a1 + 48);
  return (*(__int64 (__fastcall **)(_QWORD *))(*v1 + 80LL))(v1) + v1[4] - v1[2];
}
