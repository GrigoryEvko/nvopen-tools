// Function: sub_1F97190
// Address: 0x1f97190
//
unsigned __int64 __fastcall sub_1F97190(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // rsi
  __int64 *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned int v13; // edx
  unsigned int v14; // r8d
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v19; // [rsp-10h] [rbp-70h]
  __int64 v20; // [rsp+0h] [rbp-60h] BYREF
  int v21; // [rsp+8h] [rbp-58h]
  __int64 (__fastcall **v22)(); // [rsp+10h] [rbp-50h] BYREF
  __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 *v25; // [rsp+28h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 72);
  v20 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v20, v9, 2);
  v10 = (__int64 *)*a1;
  v21 = *(_DWORD *)(a2 + 64);
  v11 = sub_1D309E0(
          v10,
          145,
          (__int64)&v20,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          a4,
          a5,
          a6,
          (unsigned __int64)a3);
  v25 = a1;
  v12 = v11;
  v14 = v13;
  v15 = *(_QWORD *)(*a1 + 664);
  v24 = *a1;
  v23 = v15;
  *(_QWORD *)(v24 + 664) = &v22;
  v16 = *a1;
  v22 = off_49FFF30;
  sub_1D44C70(v16, a2, 0, v11, v14);
  sub_1D44C70(*a1, a2, 1, a3, 1u);
  sub_1F81E80(a1, a2);
  sub_1F81BC0((__int64)a1, v12);
  v17 = v20;
  *(_QWORD *)(v24 + 664) = v23;
  result = v19;
  if ( v17 )
    return sub_161E7C0((__int64)&v20, v17);
  return result;
}
