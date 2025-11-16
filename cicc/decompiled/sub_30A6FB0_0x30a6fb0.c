// Function: sub_30A6FB0
// Address: 0x30a6fb0
//
__int64 __fastcall sub_30A6FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  void (__fastcall *v8)(__int64 (__fastcall ***)(__int64), __int64); // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 result; // rax
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 (__fastcall **v19[2])(__int64); // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v20)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-30h]
  __int64 (__fastcall *v21)(__int64 (__fastcall ***)(__int64), __int64); // [rsp+28h] [rbp-28h]

  v6 = a1 + 8;
  v7 = *(_QWORD *)(a1 + 24);
  v18 = a3;
  v19[0] = (__int64 (__fastcall **)(__int64))&v17;
  v8 = (void (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64))sub_30A6DE0;
  v17 = a2;
  v9 = v7 + 40;
  v19[1] = (__int64 (__fastcall **)(__int64))v19;
  v21 = sub_30A6DE0;
  v20 = sub_30A68C0;
  if ( v7 == a1 + 8 )
    return ((__int64 (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64 (__fastcall ***)(__int64), __int64, __int64, __int64, __int64, __int64, __int64))sub_30A68C0)(
             v19,
             v19,
             3,
             a4,
             a5,
             a6,
             v17,
             v18);
  while ( 1 )
  {
    v8(v19, v9);
    v10 = v7;
    v11 = sub_220EF30(v7);
    v7 = v11;
    if ( v6 == v11 )
      break;
    v9 = v11 + 40;
    if ( !v20 )
      sub_4263D6(v10, v9, v12);
    v8 = (void (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64))v21;
  }
  result = (__int64)v20;
  if ( v20 )
    return ((__int64 (__fastcall *)(__int64 (__fastcall ***)(__int64), __int64 (__fastcall ***)(__int64), __int64, __int64, __int64, __int64, __int64, __int64))v20)(
             v19,
             v19,
             3,
             v13,
             v14,
             v15,
             v17,
             v18);
  return result;
}
