// Function: sub_1D0E0B0
// Address: 0x1d0e0b0
//
void __fastcall sub_1D0E0B0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = *a2;
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v3;
  *(_BYTE *)(a1 + 24) = 0;
  sub_1D0DF70(a1);
  sub_1D0DFF0(a1);
}
