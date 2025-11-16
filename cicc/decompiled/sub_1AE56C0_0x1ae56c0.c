// Function: sub_1AE56C0
// Address: 0x1ae56c0
//
__int64 __fastcall sub_1AE56C0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21[2]; // [rsp+0h] [rbp-40h] BYREF
  char v22; // [rsp+10h] [rbp-30h]
  char v23; // [rsp+11h] [rbp-2Fh]

  v21[0] = sub_16498A0((__int64)a2);
  v12 = sub_161BE60(v21, 1u, 0x7D0u);
  v15 = sub_1AA92B0(a3, (__int64)a2, 0, v12, *(_QWORD *)(a1 + 8), 0, a4, a5, a6, a7, v13, v14, a10, a11)[5];
  v23 = 1;
  v21[0] = (__int64)"cdce.call";
  v22 = 3;
  sub_164B780(v15, v21);
  v16 = sub_157F1C0(v15);
  v23 = 1;
  v22 = 3;
  v21[0] = (__int64)"cdce.end";
  sub_164B780(v16, v21);
  sub_15F2070(a2);
  v17 = (__int64 *)sub_157EE30(v15);
  sub_157E9D0(v15 + 40, (__int64)a2);
  v18 = *v17;
  v19 = a2[3];
  a2[4] = v17;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  a2[3] = v18 | v19 & 7;
  *(_QWORD *)(v18 + 8) = a2 + 3;
  result = *v17 & 7;
  *v17 = result | (unsigned __int64)(a2 + 3);
  return result;
}
