// Function: sub_23B3D30
// Address: 0x23b3d30
//
__int64 __fastcall sub_23B3D30(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 33) = a2;
  *(_QWORD *)a1 = &unk_4A15D90;
  result = sub_C5F790(a1, a2);
  *(_QWORD *)(a1 + 40) = result;
  return result;
}
