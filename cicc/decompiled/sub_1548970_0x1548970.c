// Function: sub_1548970
// Address: 0x1548970
//
__int64 __fastcall sub_1548970(__int64 a1)
{
  _QWORD *v1; // rbx

  v1 = *(_QWORD **)(a1 + 40);
  return (*(__int64 (__fastcall **)(_QWORD *))(*v1 + 64LL))(v1) + v1[3] - v1[1];
}
