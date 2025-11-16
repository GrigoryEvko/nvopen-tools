// Function: sub_206C980
// Address: 0x206c980
//
__int64 *__fastcall sub_206C980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6, __m128i a7)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 *result; // rax
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 *v16; // r14
  unsigned __int8 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  const void **v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rbx
  int v23; // edx
  int v24; // r14d
  __int64 *v25; // rax
  __int64 v26; // rsi
  __int128 v27; // [rsp-10h] [rbp-90h]
  __int64 *v28; // [rsp-10h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  const void **v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+38h] [rbp-48h] BYREF
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  int v33; // [rsp+48h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v8 = *(_QWORD *)(a2 - 8);
  }
  else
  {
    a3 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v8 = a2 - a3;
  }
  if ( *(_BYTE *)(*(_QWORD *)v8 + 16LL) > 0x10u )
    return sub_206C720(a1, a2, 0x4Du, a5, a6, a7);
  v9 = *(_QWORD *)v8;
  if ( v9 != sub_15A14C0(*(_QWORD *)a2, a2, a3, a4) )
    return sub_206C720(a1, a2, 0x4Du, a5, a6, a7);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v11 = *(_QWORD *)(a2 - 8);
  else
    v11 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v12 = sub_20685E0(a1, *(__int64 **)(v11 + 24), a5, a6, a7);
  v13 = *(__int64 **)(a1 + 552);
  v15 = v14;
  v16 = v12;
  v17 = (unsigned __int8 *)(v12[5] + 16LL * (unsigned int)v14);
  v18 = *(_QWORD *)a1;
  v19 = *v17;
  v20 = (const void **)*((_QWORD *)v17 + 1);
  v32 = 0;
  v33 = *(_DWORD *)(a1 + 536);
  if ( v18 )
  {
    if ( &v32 != (__int64 *)(v18 + 48) )
    {
      v21 = *(_QWORD *)(v18 + 48);
      v32 = v21;
      if ( v21 )
      {
        v29 = v19;
        v30 = v20;
        sub_1623A60((__int64)&v32, v21, 2);
        v19 = v29;
        v20 = v30;
      }
    }
  }
  *((_QWORD *)&v27 + 1) = v15;
  *(_QWORD *)&v27 = v16;
  v31 = a2;
  v22 = sub_1D309E0(
          v13,
          162,
          (__int64)&v32,
          v19,
          v20,
          0,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64,
          v27);
  v24 = v23;
  v25 = sub_205F5C0(a1 + 8, &v31);
  v26 = v32;
  v25[1] = v22;
  *((_DWORD *)v25 + 4) = v24;
  result = v28;
  if ( v26 )
    return (__int64 *)sub_161E7C0((__int64)&v32, v26);
  return result;
}
