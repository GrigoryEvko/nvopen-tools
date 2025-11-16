// Function: sub_377DC10
// Address: 0x377dc10
//
void __fastcall sub_377DC10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 v16; // r8
  int v17; // edx
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rsi
  int v21; // edx
  __int128 v22; // [rsp-10h] [rbp-E0h]
  __int64 v23; // [rsp+30h] [rbp-A0h] BYREF
  int v24; // [rsp+38h] [rbp-98h]
  unsigned int v25; // [rsp+40h] [rbp-90h] BYREF
  __int64 v26; // [rsp+48h] [rbp-88h]
  unsigned int v27; // [rsp+50h] [rbp-80h]
  __int64 v28; // [rsp+58h] [rbp-78h]
  __m128i v29[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v30; // [rsp+80h] [rbp-50h] BYREF
  __int64 v31; // [rsp+90h] [rbp-40h]
  __int64 v32; // [rsp+98h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v23 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v23, v8, 1);
  v9 = a1[1];
  v24 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v30.m128i_i16[0] = v11;
  v30.m128i_i64[1] = v12;
  sub_33D0340((__int64)&v25, v9, v30.m128i_i64);
  sub_3777990(v29, a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a5);
  sub_3408380(
    &v30,
    (_QWORD *)a1[1],
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
    **(unsigned __int16 **)(a2 + 48),
    *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
    a5,
    (__int64)&v23);
  v14 = sub_340F900(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v23,
          v25,
          v26,
          v13,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)v29,
          *(_OWORD *)&v30);
  *((_QWORD *)&v22 + 1) = v32;
  v15 = v27;
  *(_QWORD *)a3 = v14;
  v16 = v28;
  *(_QWORD *)&v22 = v31;
  *(_DWORD *)(a3 + 8) = v17;
  v19 = sub_340F900(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v23,
          v15,
          v16,
          v18,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          *(_OWORD *)&v29[1],
          v22);
  v20 = v23;
  *(_QWORD *)a4 = v19;
  *(_DWORD *)(a4 + 8) = v21;
  if ( v20 )
    sub_B91220((__int64)&v23, v20);
}
