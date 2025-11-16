// Function: sub_B4D5A0
// Address: 0xb4d5a0
//
__int64 __fastcall sub_B4D5A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int16 a6,
        __int16 a7,
        char a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v14; // rax
  __int64 v15; // r11
  __int64 *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  _QWORD v21[8]; // [rsp+20h] [rbp-40h] BYREF

  v14 = sub_BD5C60(a3, a2);
  v15 = sub_BCB2A0(v14);
  v16 = *(__int64 **)(a3 + 8);
  v17 = *v16;
  v21[1] = v15;
  v21[0] = v16;
  v18 = sub_BD0B90(v17, v21, 2, 0);
  sub_B44260(a1, v18, 36, 3u, a9, a10);
  return sub_B4D470(a1, a2, a3, a4, a5, a6, a7, a8);
}
