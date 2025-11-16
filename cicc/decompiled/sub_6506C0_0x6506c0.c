// Function: sub_6506C0
// Address: 0x6506c0
//
__int64 __fastcall sub_6506C0(__int64 a1, _QWORD *a2, unsigned int a3)
{
  __int64 v4; // r14
  __int64 v5; // r12
  char v6; // al
  _BYTE v8[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = sub_87D510(a1, v8);
  v5 = sub_726DD0();
  v6 = v8[0];
  *(_QWORD *)(v5 + 24) = v4;
  *(_BYTE *)(v5 + 16) = v6;
  *(_QWORD *)(v5 + 8) = *a2;
  sub_733230(v5, a3);
  return v5;
}
