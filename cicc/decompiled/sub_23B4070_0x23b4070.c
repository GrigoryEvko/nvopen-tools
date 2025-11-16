// Function: sub_23B4070
// Address: 0x23b4070
//
__int64 __fastcall sub_23B4070(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 33) = a2;
  *(_QWORD *)a1 = &unk_4A15E40;
  result = sub_C5F790(a1, a2);
  *(_QWORD *)(a1 + 40) = result;
  return result;
}
