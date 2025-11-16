// Function: sub_29FE890
// Address: 0x29fe890
//
void __fastcall sub_29FE890(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __m128i *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  __int64 v16; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v18; // [rsp+28h] [rbp-58h]
  __int64 v19; // [rsp+30h] [rbp-50h]
  __int64 v20; // [rsp+38h] [rbp-48h]
  __int64 v21; // [rsp+40h] [rbp-40h]

  v1 = (__int64 *)sub_AA48A0(**(_QWORD **)(a1 + 32));
  v14 = sub_B9C770(v1, 0, 0, 0, 1);
  v17 = sub_B9B140(v1, "llvm.loop.unroll.disable", 0x18u);
  v15 = sub_B9C770(v1, &v17, (__int64 *)1, 0, 1);
  v2 = sub_BCB2A0(v1);
  v3 = sub_ACD640(v2, 0, 0);
  v4 = sub_B98A20(v3, 0);
  v5 = sub_B9B140(v1, "llvm.loop.vectorize.enable", 0x1Au);
  v18 = v4;
  v17 = v5;
  v16 = sub_B9C770(v1, &v17, (__int64 *)2, 0, 1);
  v17 = sub_B9B140(v1, "llvm.loop.licm_versioning.disable", 0x21u);
  v6 = sub_B9C770(v1, &v17, (__int64 *)1, 0, 1);
  v7 = sub_B9B140(v1, "llvm.loop.distribute.enable", 0x1Bu);
  v18 = v4;
  v17 = v7;
  v8 = sub_B9C770(v1, &v17, (__int64 *)2, 0, 1);
  v17 = v14;
  v18 = (_QWORD *)v15;
  v19 = v16;
  v20 = v6;
  v21 = v8;
  v9 = (__m128i *)sub_B9C770(v1, &v17, (__int64 *)5, 0, 1);
  sub_BA6610(v9, 0, (unsigned __int8 *)v9);
  sub_D49440(a1, (__int64)v9, v10, v11, v12, v13);
}
