// Function: sub_2AC29F0
// Address: 0x2ac29f0
//
void __fastcall sub_2AC29F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-3A8h] BYREF
  __m128i v17; // [rsp+10h] [rbp-3A0h] BYREF
  _QWORD v18[10]; // [rsp+20h] [rbp-390h] BYREF
  _BYTE v19[344]; // [rsp+70h] [rbp-340h] BYREF
  __int64 v20; // [rsp+1C8h] [rbp-1E8h]
  _QWORD v21[10]; // [rsp+1D0h] [rbp-1E0h] BYREF
  _BYTE v22[344]; // [rsp+220h] [rbp-190h] BYREF
  __int64 v23; // [rsp+378h] [rbp-38h]

  v2 = *a1;
  v3 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v3)
    || (v14 = sub_B2BE50(v2),
        v15 = sub_B6F970(v14),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15)) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    v9 = **(_QWORD **)(v8 + 32);
    sub_D4BD20(&v16, v8, v4, v5, v6, v7);
    sub_B157E0((__int64)&v17, &v16);
    sub_B17850((__int64)v21, (__int64)"loop-vectorize", (__int64)"VectorizationCodeSize", 21, &v17, v9);
    sub_B18290(
      (__int64)v21,
      "Code-size may be reduced by not forcing vectorization, or by source-code modifications eliminating the need for ru"
      "ntime checks (e.g., adding 'restrict').",
      0x99u);
    sub_23FE290((__int64)v18, (__int64)v21, v10, v11, v12, v13);
    v20 = v23;
    v18[0] = &unk_49D9DE8;
    v21[0] = &unk_49D9D40;
    sub_23FD590((__int64)v22);
    sub_9C6650(&v16);
    sub_1049740(a1, (__int64)v18);
    v18[0] = &unk_49D9D40;
    sub_23FD590((__int64)v19);
  }
}
