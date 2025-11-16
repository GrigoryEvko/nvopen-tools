// Function: sub_206EF80
// Address: 0x206ef80
//
unsigned __int64 __fastcall sub_206EF80(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 **v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  const void **v14; // rdx
  __int64 *v15; // r10
  const void **v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  int v21; // edx
  int v22; // r13d
  __int64 *v23; // rax
  __int64 v24; // rsi
  unsigned __int64 result; // rax
  __int128 v26; // [rsp-10h] [rbp-90h]
  unsigned __int64 v27; // [rsp-10h] [rbp-90h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  const void **v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  __int64 *v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+38h] [rbp-48h] BYREF
  __int64 v33; // [rsp+40h] [rbp-40h] BYREF
  int v34; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 ***)(a2 - 8);
  else
    v6 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = sub_20685E0(a1, *v6, a3, a4, a5);
  v8 = *(_QWORD *)(a1 + 552);
  v10 = v9;
  v11 = *(_QWORD *)(v8 + 16);
  v30 = *(_QWORD *)a2;
  v12 = sub_1E0A0C0(*(_QWORD *)(v8 + 32));
  LOBYTE(v13) = sub_204D4D0(v11, v12, v30);
  v33 = 0;
  v15 = *(__int64 **)(a1 + 552);
  v16 = v14;
  v17 = v13;
  v18 = *(_QWORD *)a1;
  v34 = *(_DWORD *)(a1 + 536);
  if ( v18 )
  {
    if ( &v33 != (__int64 *)(v18 + 48) )
    {
      v19 = *(_QWORD *)(v18 + 48);
      v33 = v19;
      if ( v19 )
      {
        v28 = v17;
        v29 = v14;
        v31 = v15;
        sub_1623A60((__int64)&v33, v19, 2);
        v17 = v28;
        v16 = v29;
        v15 = v31;
      }
    }
  }
  *((_QWORD *)&v26 + 1) = v10;
  *(_QWORD *)&v26 = v7;
  v32 = a2;
  v20 = sub_1D309E0(
          v15,
          147,
          (__int64)&v33,
          v17,
          v16,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          v26);
  v22 = v21;
  v23 = sub_205F5C0(a1 + 8, &v32);
  v24 = v33;
  v23[1] = v20;
  *((_DWORD *)v23 + 4) = v22;
  result = v27;
  if ( v24 )
    return sub_161E7C0((__int64)&v33, v24);
  return result;
}
