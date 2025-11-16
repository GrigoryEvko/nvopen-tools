// Function: sub_211A3C0
// Address: 0x211a3c0
//
__int64 __fastcall sub_211A3C0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r13
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 *v15; // rax
  __int64 v16; // rax
  double v17; // xmm4_8
  double v18; // xmm5_8
  _QWORD v20[8]; // [rsp+10h] [rbp-40h] BYREF

  v10 = *(__int64 **)(a2 + 40);
  v11 = sub_1646BA0(*(__int64 **)(a1 + 176), 0);
  v12 = (__int64 *)sub_1643270((_QWORD *)*v10);
  v20[0] = v11;
  v13 = sub_1644EA0(v12, v20, 1, 0);
  *(_QWORD *)(a1 + 184) = sub_1632080((__int64)v10, (__int64)"_Unwind_SjLj_Register", 21, v13, 0);
  v14 = sub_1646BA0(*(__int64 **)(a1 + 176), 0);
  v15 = (__int64 *)sub_1643270((_QWORD *)*v10);
  v20[0] = v14;
  v16 = sub_1644EA0(v15, v20, 1, 0);
  *(_QWORD *)(a1 + 192) = sub_1632080((__int64)v10, (__int64)"_Unwind_SjLj_Unregister", 23, v16, 0);
  *(_QWORD *)(a1 + 208) = sub_15E26F0(v10, 101, 0, 0);
  *(_QWORD *)(a1 + 216) = sub_15E26F0(v10, 202, 0, 0);
  *(_QWORD *)(a1 + 224) = sub_15E26F0(v10, 201, 0, 0);
  *(_QWORD *)(a1 + 200) = sub_15E26F0(v10, 51, 0, 0);
  *(_QWORD *)(a1 + 232) = sub_15E26F0(v10, 49, 0, 0);
  *(_QWORD *)(a1 + 240) = sub_15E26F0(v10, 46, 0, 0);
  *(_QWORD *)(a1 + 248) = sub_15E26F0(v10, 47, 0, 0);
  return sub_21174B0(a1, a2, a3, a4, a5, a6, v17, v18, a9, a10);
}
