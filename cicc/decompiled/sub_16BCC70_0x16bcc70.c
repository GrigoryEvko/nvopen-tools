// Function: sub_16BCC70
// Address: 0x16bcc70
//
__int64 __fastcall sub_16BCC70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 result; // rax

  v7 = a1 + 8;
  *(_QWORD *)(v7 - 8) = &unk_49EF2C0;
  result = sub_16E2FC0(v7, a2);
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 48) = a4;
  return result;
}
