// Function: sub_70DAE0
// Address: 0x70dae0
//
__int64 __fastcall sub_70DAE0(__int64 a1, int a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  char v4; // al
  __int64 v5; // r8

  v2 = *(_QWORD *)(a1 + 128);
  sub_724A80(a1, 7);
  v3 = sub_8D4870(v2);
  v4 = sub_8D2310(v3);
  *(_QWORD *)(a1 + 200) = 0;
  *(_BYTE *)(a1 + 192) = (2 * (v4 & 1)) | *(_BYTE *)(a1 + 192) & 0xFD;
  return sub_70C9E0(a1, v2, a2, 2 * (v4 & 1u), v5);
}
