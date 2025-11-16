// Function: sub_D4A9E0
// Address: 0xd4a9e0
//
void __fastcall sub_D4A9E0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __m128i *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  __int64 v14[6]; // [rsp+10h] [rbp-30h] BYREF

  v1 = (__int64 *)sub_AA48A0(**(_QWORD **)(a1 + 32));
  v14[0] = sub_B9B140(v1, "llvm.loop.unroll.disable", 0x18u);
  v2 = sub_B9C770(v1, v14, (__int64 *)1, 0, 1);
  v7 = sub_D49300(a1, (__int64)v14, v3, v4, v5, v6);
  v13 = v2;
  v14[0] = (__int64)"llvm.loop.unroll.";
  v14[1] = 17;
  v8 = sub_D4A520(v1, v7, (__int64)v14, 1, (__int64)&v13, 1);
  sub_D49440(a1, (__int64)v8, v9, v10, v11, v12);
}
