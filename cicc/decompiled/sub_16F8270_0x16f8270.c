// Function: sub_16F8270
// Address: 0x16f8270
//
_QWORD *__fastcall sub_16F8270(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // r10
  unsigned __int64 v6[2]; // [rsp+8h] [rbp-10h] BYREF

  v3 = *a1;
  v4 = *(_QWORD *)(a2 + 16);
  v6[1] = *(_QWORD *)(a2 + 24);
  v6[0] = v4;
  return sub_16D14E0(*(__int64 **)v3, v4, 0, a3, v6, 1, 0, 0, *(_BYTE *)(v3 + 75));
}
