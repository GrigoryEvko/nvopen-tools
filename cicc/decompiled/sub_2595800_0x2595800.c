// Function: sub_2595800
// Address: 0x2595800
//
__int64 __fastcall sub_2595800(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // [rsp-10h] [rbp-30h]
  unsigned __int64 v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v2 = sub_25096F0((_QWORD *)(a1 + 72));
  sub_250D230(v5, v2, 4, 0);
  sub_25952D0(a2, v5[0], v5[1], a1, 0, 0, 1);
  return v4;
}
