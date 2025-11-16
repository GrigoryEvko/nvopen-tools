// Function: sub_381FF10
// Address: 0x381ff10
//
void __fastcall sub_381FF10(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v11; // rax
  unsigned __int16 v12; // si
  __int64 v13; // r8
  __int64 v14; // rax
  __int32 v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // eax
  __m128i v19; // xmm0
  _QWORD *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // rax
  int v24; // r9d
  __int64 v25; // rcx
  int v26; // edx
  unsigned int v27; // edx
  __int128 v28; // rax
  __int64 v29; // r9
  int v30; // edx
  __int64 v31; // rdx
  __int128 v32; // [rsp-10h] [rbp-C0h]
  unsigned __int64 v33; // [rsp+0h] [rbp-B0h]
  int v34; // [rsp+8h] [rbp-A8h]
  _QWORD *v35; // [rsp+8h] [rbp-A8h]
  __int64 v36; // [rsp+30h] [rbp-80h] BYREF
  int v37; // [rsp+38h] [rbp-78h]
  __m128i v38; // [rsp+40h] [rbp-70h] BYREF
  __int64 v39; // [rsp+50h] [rbp-60h]
  __int64 v40; // [rsp+58h] [rbp-58h]
  __m128i v41; // [rsp+60h] [rbp-50h] BYREF
  __int64 v42; // [rsp+70h] [rbp-40h]
  __int64 v43; // [rsp+78h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v36 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v36, v8, 1);
  v9 = *a1;
  v37 = *(_DWORD *)(a2 + 72);
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v14 = a1[1];
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v41, v9, *(_QWORD *)(v14 + 64), v12, v13);
    LOWORD(v15) = v41.m128i_i16[4];
    v38.m128i_i16[0] = v41.m128i_i16[4];
    v38.m128i_i64[1] = v42;
  }
  else
  {
    v15 = v10(v9, *(_QWORD *)(v14 + 64), v12, v13);
    v38.m128i_i32[0] = v15;
    v38.m128i_i64[1] = v31;
  }
  if ( (_WORD)v15 )
  {
    if ( (_WORD)v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      BUG();
    v17 = 16LL * ((unsigned __int16)v15 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v17];
    LOBYTE(v17) = byte_444C4A0[v17 + 8];
  }
  else
  {
    v16 = sub_3007260((__int64)&v38);
    v39 = v16;
    v40 = v17;
  }
  v41.m128i_i8[8] = v17;
  v41.m128i_i64[0] = v16;
  v18 = sub_CA1930(&v41);
  v19 = _mm_loadu_si128(&v38);
  *((_QWORD *)&v32 + 1) = 1;
  v20 = (_QWORD *)a1[1];
  v34 = v18;
  v21 = *(_QWORD *)(a2 + 40);
  LOWORD(v42) = 1;
  *(_QWORD *)&v32 = v21;
  v41 = v19;
  v43 = 0;
  v23 = sub_3411BE0(v20, 0xE7u, (__int64)&v36, (unsigned __int16 *)&v41, 2, v22, v32);
  v24 = v34;
  v25 = v38.m128i_i64[1];
  LODWORD(v20) = v26;
  *(_QWORD *)a3 = v23;
  v27 = v38.m128i_i32[0];
  v33 = (unsigned __int64)v23;
  *(_DWORD *)(a3 + 8) = (_DWORD)v20;
  v35 = (_QWORD *)a1[1];
  *(_QWORD *)&v28 = sub_3400E40((__int64)v35, (unsigned int)(v24 - 1), v27, v25, (__int64)&v36, v19);
  *(_QWORD *)a4 = sub_3406EB0(v35, 0xBFu, (__int64)&v36, v38.m128i_u32[0], v38.m128i_i64[1], v29, *(_OWORD *)a3, v28);
  *(_DWORD *)(a4 + 8) = v30;
  sub_3760E70((__int64)a1, a2, 1, v33, 1);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
}
