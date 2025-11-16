// Function: sub_303D290
// Address: 0x303d290
//
__int64 __fastcall sub_303D290(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r8
  int v5; // r9d
  const __m128i *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r15
  unsigned int v12; // edx
  unsigned __int64 v13; // r13
  unsigned int v14; // edx
  __int128 v15; // rax
  __int64 v16; // r12
  __int128 v18; // [rsp-20h] [rbp-A0h]
  __int128 v19; // [rsp-20h] [rbp-A0h]
  __int128 v20; // [rsp-10h] [rbp-90h]
  __int128 v21; // [rsp-10h] [rbp-90h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  int v24; // [rsp+8h] [rbp-78h]
  __int128 v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+20h] [rbp-60h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  int v29; // [rsp+48h] [rbp-38h]

  v4 = a2;
  v5 = a4;
  v6 = *(const __m128i **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = v6[2].m128i_i64[1];
  v28 = v7;
  v9 = v6[3].m128i_i64[0];
  v10 = v6[5].m128i_i64[0];
  v11 = v6[5].m128i_i64[1];
  v25 = (__int128)_mm_loadu_si128(v6);
  if ( v7 )
  {
    v23 = v4;
    sub_B96E90((__int64)&v28, v7, 1);
    v5 = a4;
    v4 = v23;
  }
  *((_QWORD *)&v20 + 1) = v9;
  *(_QWORD *)&v20 = v8;
  v24 = v5;
  v29 = *(_DWORD *)(v4 + 72);
  *((_QWORD *)&v18 + 1) = v11;
  *(_QWORD *)&v18 = v10;
  v27 = sub_33FAF80(v5, 215, (unsigned int)&v28, 7, 0, v5, v20);
  v13 = v12 | v9 & 0xFFFFFFFF00000000LL;
  v26 = sub_33FAF80(v24, 215, (unsigned int)&v28, 7, 0, v24, v18);
  *((_QWORD *)&v21 + 1) = v14 | v11 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v21 = v26;
  *((_QWORD *)&v19 + 1) = v13;
  *(_QWORD *)&v19 = v27;
  *(_QWORD *)&v15 = sub_340F900(v24, 205, (unsigned int)&v28, 7, 0, v24, v25, v19, v21);
  v16 = sub_33FAF80(v24, 216, (unsigned int)&v28, 2, 0, v24, v15);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v16;
}
