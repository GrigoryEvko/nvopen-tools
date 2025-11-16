// Function: sub_1F6F410
// Address: 0x1f6f410
//
__int64 *__fastcall sub_1F6F410(__int64 **a1, __int64 a2)
{
  const __m128i *v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 *v8; // r9
  __int64 v9; // rsi
  const void ***v10; // r10
  int v11; // r8d
  __int64 *result; // rax
  __int128 v13; // [rsp-20h] [rbp-90h]
  __int128 v14; // [rsp-10h] [rbp-80h]
  const void ***v15; // [rsp+0h] [rbp-70h]
  int v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  __int64 *v19; // [rsp+18h] [rbp-58h]
  __int128 v20; // [rsp+20h] [rbp-50h]
  __int64 *v21; // [rsp+20h] [rbp-50h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  int v23; // [rsp+38h] [rbp-38h]

  v2 = *(const __m128i **)(a2 + 32);
  v3 = v2[2].m128i_i64[1];
  v4 = v2[3].m128i_i64[0];
  v5 = v2[7].m128i_i64[1];
  v20 = (__int128)_mm_loadu_si128(v2);
  v6 = v2[8].m128i_i64[0];
  if ( !sub_1D185B0(v2[5].m128i_i64[0]) )
    return 0;
  v7 = a2;
  v8 = *a1;
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *(const void ****)(a2 + 40);
  v11 = *(_DWORD *)(a2 + 60);
  v22 = v9;
  if ( v9 )
  {
    v15 = v10;
    v16 = v11;
    v17 = a2;
    v19 = v8;
    sub_1623A60((__int64)&v22, v9, 2);
    v10 = v15;
    v11 = v16;
    v7 = v17;
    v8 = v19;
  }
  *((_QWORD *)&v14 + 1) = v6;
  *(_QWORD *)&v14 = v5;
  *((_QWORD *)&v13 + 1) = v4;
  *(_QWORD *)&v13 = v3;
  v23 = *(_DWORD *)(v7 + 64);
  result = sub_1D37470(v8, 137, (__int64)&v22, v10, v11, (__int64)v8, v20, v13, v14);
  if ( v22 )
  {
    v21 = result;
    sub_161E7C0((__int64)&v22, v22);
    return v21;
  }
  return result;
}
