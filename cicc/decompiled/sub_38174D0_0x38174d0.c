// Function: sub_38174D0
// Address: 0x38174d0
//
__int64 __fastcall sub_38174D0(__int64 a1, __int64 a2)
{
  const __m128i *v2; // rax
  unsigned __int16 *v3; // rdx
  __int64 v4; // r12
  unsigned int v5; // r14d
  __int64 v6; // r13
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r9
  __int64 v10; // r8
  unsigned __int8 *v11; // r10
  __int64 v12; // r11
  __int64 v13; // r12
  __int128 v15; // [rsp-20h] [rbp-90h]
  __int128 v16; // [rsp-10h] [rbp-80h]
  unsigned __int8 *v17; // [rsp+0h] [rbp-70h]
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  _QWORD *v21; // [rsp+18h] [rbp-58h]
  __int128 v22; // [rsp+20h] [rbp-50h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  int v24; // [rsp+38h] [rbp-38h]

  v2 = *(const __m128i **)(a2 + 40);
  v3 = (unsigned __int16 *)(*(_QWORD *)(v2->m128i_i64[0] + 48) + 16LL * v2->m128i_u32[2]);
  v4 = v2[5].m128i_i64[0];
  v5 = *v3;
  v22 = (__int128)_mm_loadu_si128(v2);
  v6 = v2[5].m128i_i64[1];
  v20 = *((_QWORD *)v3 + 1);
  v7 = sub_375B580(a1, v2[2].m128i_i64[1], (__m128i)v22, v2[3].m128i_i64[0], *v3, v20);
  v9 = *(_QWORD **)(a1 + 8);
  v10 = v20;
  v11 = v7;
  v12 = v8;
  v23 = *(_QWORD *)(a2 + 80);
  if ( v23 )
  {
    v18 = v8;
    v19 = v20;
    v21 = v9;
    v17 = v7;
    sub_B96E90((__int64)&v23, v23, 1);
    v11 = v17;
    v12 = v18;
    v10 = v19;
    v9 = v21;
  }
  *((_QWORD *)&v16 + 1) = v6;
  *(_QWORD *)&v16 = v4;
  *((_QWORD *)&v15 + 1) = v12;
  *(_QWORD *)&v15 = v11;
  v24 = *(_DWORD *)(a2 + 72);
  v13 = sub_340F900(v9, 0xABu, (__int64)&v23, v5, v10, (__int64)v9, v22, v15, v16);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v13;
}
