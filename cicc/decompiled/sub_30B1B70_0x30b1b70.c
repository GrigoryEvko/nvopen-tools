// Function: sub_30B1B70
// Address: 0x30b1b70
//
__int64 *__fastcall sub_30B1B70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __m128i v15; // [rsp+0h] [rbp-50h] BYREF
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  v6 = *(__int64 **)(a3 + 32);
  v7 = a5[3];
  v8 = a5[4];
  v9 = *a5;
  v10 = *v6;
  v16 = v7;
  v15.m128i_i64[0] = v9;
  v11 = *(_QWORD *)(v10 + 72);
  v15.m128i_i64[1] = v8;
  v17 = v11;
  v12 = sub_22077B0(0xE0u);
  v13 = v12;
  if ( v12 )
    sub_30B1350(v12, a3, v7, &v15);
  *a1 = v13;
  return a1;
}
