// Function: sub_37191E0
// Address: 0x37191e0
//
__int64 __fastcall sub_37191E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 result; // rax

  v3 = a1 + 8;
  *(_QWORD *)(v3 - 8) = &unk_4A352E0;
  result = sub_12556C0(v3, a2);
  *(_QWORD *)(a1 + 56) = 0;
  return result;
}
