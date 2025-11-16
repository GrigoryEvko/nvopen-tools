// Function: sub_1E1C1F0
// Address: 0x1e1c1f0
//
__int64 __fastcall sub_1E1C1F0(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 *a3,
        __int64 a4,
        char a5,
        __int32 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned __int64 v11; // rdx
  __int64 v12; // rax

  v8 = *(_QWORD *)(a1 + 56);
  sub_1E1C0A0(v8, a3, a4, a5, a6, a7, a8);
  v10 = v9;
  sub_1DD5BA0((__int64 *)(a1 + 16), v9);
  v11 = *a2;
  v12 = *(_QWORD *)v10;
  *(_QWORD *)(v10 + 8) = a2;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v10 = v11 | v12 & 7;
  *(_QWORD *)(v11 + 8) = v10;
  *a2 = v10 | *a2 & 7;
  return v8;
}
