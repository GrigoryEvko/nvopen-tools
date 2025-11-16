// Function: sub_C63E60
// Address: 0xc63e60
//
__int64 __fastcall sub_C63E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 result; // rax

  v6 = a1 + 8;
  *(_QWORD *)(v6 - 8) = &unk_49DC7F0;
  result = sub_CA0F50(v6, a4);
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 48) = a3;
  *(_BYTE *)(a1 + 56) = 0;
  return result;
}
