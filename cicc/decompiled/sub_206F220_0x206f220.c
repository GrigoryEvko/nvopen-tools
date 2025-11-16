// Function: sub_206F220
// Address: 0x206f220
//
__int64 *__fastcall sub_206F220(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  const void **v15; // rdx
  __int64 *v16; // r15
  const void **v17; // r9
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r13
  int v22; // edx
  int v23; // r15d
  __int64 *result; // rax
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h]
  const void **v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  __int64 v29; // [rsp+38h] [rbp-48h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h] BYREF
  int v31; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v8 = *(_QWORD *)a2;
  v9 = (__int64)v7;
  v10 = *(_QWORD *)(a1 + 552);
  v28 = v11;
  v12 = *(_QWORD *)(v10 + 16);
  v13 = sub_1E0A0C0(*(_QWORD *)(v10 + 32));
  LOBYTE(v14) = sub_204D4D0(v12, v13, v8);
  v30 = 0;
  v16 = *(__int64 **)(a1 + 552);
  v17 = v15;
  v18 = v14;
  v19 = *(_QWORD *)a1;
  v31 = *(_DWORD *)(a1 + 536);
  if ( v19 )
  {
    if ( &v30 != (__int64 *)(v19 + 48) )
    {
      v20 = *(_QWORD *)(v19 + 48);
      v30 = v20;
      if ( v20 )
      {
        v26 = v18;
        v27 = v15;
        sub_1623A60((__int64)&v30, v20, 2);
        v18 = v26;
        v17 = v27;
      }
    }
  }
  v29 = a2;
  v21 = sub_1D323C0(
          v16,
          v9,
          v28,
          (__int64)&v30,
          v18,
          v17,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64);
  v23 = v22;
  result = sub_205F5C0(a1 + 8, &v29);
  v25 = v30;
  result[1] = v21;
  *((_DWORD *)result + 4) = v23;
  if ( v25 )
    return (__int64 *)sub_161E7C0((__int64)&v30, v25);
  return result;
}
