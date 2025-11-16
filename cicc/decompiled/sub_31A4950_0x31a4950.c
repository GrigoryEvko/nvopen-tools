// Function: sub_31A4950
// Address: 0x31a4950
//
void __fastcall sub_31A4950(__int64 a1)
{
  __int64 *v1; // r12
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i *v13; // r12
  __int64 v14; // [rsp+0h] [rbp-110h]
  __int64 v15; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v16[2]; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v17; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v18[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v19; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD v20[4]; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int64 v21; // [rsp+80h] [rbp-90h] BYREF
  __int64 v22; // [rsp+88h] [rbp-88h]
  char *v23; // [rsp+90h] [rbp-80h]
  __int16 v24; // [rsp+A0h] [rbp-70h]
  const char *v25; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v26; // [rsp+B8h] [rbp-58h]
  const char *v27; // [rsp+C0h] [rbp-50h]
  __int16 v28; // [rsp+D0h] [rbp-40h]

  v1 = (__int64 *)sub_AA48A0(**(_QWORD **)(*(_QWORD *)(a1 + 104) + 32LL));
  LODWORD(v22) = 32;
  v25 = (const char *)sub_B9B140(v1, "llvm.loop.isvectorized", 0x16u);
  v21 = 1;
  v2 = sub_ACCFD0(v1, (__int64)&v21);
  v26 = (__int64)sub_B98A20(v2, (__int64)&v21);
  v7 = sub_B9C770(v1, (__int64 *)&v25, (__int64 *)2, 0, 1);
  if ( (unsigned int)v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  v8 = sub_D49300(*(_QWORD *)(a1 + 104), (__int64)&v25, v3, v4, v5, v6);
  v15 = v7;
  v14 = v8;
  v24 = 773;
  v23 = "vectorize.";
  v21 = (unsigned __int64)"llvm.loop.";
  v22 = 10;
  sub_CA0F50(v16, (void **)&v21);
  v28 = 773;
  v20[0] = v16[0];
  v25 = "llvm.loop.";
  v20[1] = v16[1];
  v27 = "interleave.";
  v26 = 10;
  sub_CA0F50(v18, (void **)&v25);
  v20[2] = v18[0];
  v20[3] = v18[1];
  v13 = sub_D4A520(v1, v14, (__int64)v20, 2, (__int64)&v15, 1);
  if ( (__int64 *)v18[0] != &v19 )
    j_j___libc_free_0(v18[0]);
  if ( (__int64 *)v16[0] != &v17 )
    j_j___libc_free_0(v16[0]);
  sub_D49440(*(_QWORD *)(a1 + 104), (__int64)v13, v9, v10, v11, v12);
  *(_DWORD *)(a1 + 56) = 1;
}
