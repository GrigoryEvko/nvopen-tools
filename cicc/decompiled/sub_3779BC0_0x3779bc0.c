// Function: sub_3779BC0
// Address: 0x3779bc0
//
void __fastcall sub_3779BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int128 v10; // xmm0
  __int64 v11; // r13
  int v12; // eax
  __int64 v13; // rsi
  __int16 *v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // r9
  int v19; // edx
  __int64 v20; // rax
  __int64 v21; // r15
  _QWORD *v22; // r14
  unsigned __int32 v23; // esi
  __int128 v24; // rax
  __int64 v25; // r9
  unsigned __int8 *v26; // rax
  __int64 v27; // rsi
  int v28; // edx
  __int128 v29; // [rsp-20h] [rbp-F0h]
  __int128 v30; // [rsp-20h] [rbp-F0h]
  __int64 v31; // [rsp+0h] [rbp-D0h]
  __int64 v32; // [rsp+8h] [rbp-C8h]
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v35; // [rsp+28h] [rbp-A8h]
  __int64 v36; // [rsp+50h] [rbp-80h] BYREF
  int v37; // [rsp+58h] [rbp-78h]
  __m128i v38; // [rsp+60h] [rbp-70h] BYREF
  __int64 v39[2]; // [rsp+70h] [rbp-60h] BYREF
  __m128i v40; // [rsp+80h] [rbp-50h] BYREF
  __int64 v41; // [rsp+90h] [rbp-40h]
  __int64 v42; // [rsp+98h] [rbp-38h]

  v4 = a2;
  v7 = *(__int64 **)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v7;
  v10 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 5));
  v36 = v8;
  v11 = v7[1];
  v35 = v7[5];
  if ( v8 )
  {
    v32 = v4;
    sub_B96E90((__int64)&v36, v8, 1);
    v4 = v32;
  }
  v12 = *(_DWORD *)(v4 + 72);
  v38.m128i_i64[1] = 0;
  v13 = *(_QWORD *)(a1 + 8);
  v37 = v12;
  v38.m128i_i16[0] = 0;
  v14 = *(__int16 **)(v4 + 48);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  LOWORD(v39[0]) = v15;
  v39[1] = v16;
  sub_33D0340((__int64)&v40, v13, v39);
  v17 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v29 + 1) = v11;
  *(_QWORD *)&v29 = v9;
  v33 = v41;
  v38 = _mm_loadu_si128(&v40);
  v31 = v42;
  *(_QWORD *)a3 = sub_3406EB0(v17, 0xA1u, (__int64)&v36, v40.m128i_u32[0], v38.m128i_i64[1], v18, v29, v10);
  *(_DWORD *)(a3 + 8) = v19;
  v20 = *(_QWORD *)(v35 + 96);
  if ( *(_DWORD *)(v20 + 32) <= 0x40u )
    v21 = *(_QWORD *)(v20 + 24);
  else
    v21 = **(_QWORD **)(v20 + 24);
  v22 = *(_QWORD **)(a1 + 8);
  if ( v38.m128i_i16[0] )
  {
    v23 = word_4456340[v38.m128i_u16[0] - 1];
  }
  else
  {
    v40.m128i_i64[0] = sub_3007240((__int64)&v38);
    v23 = v40.m128i_i32[0];
  }
  *(_QWORD *)&v24 = sub_3400EE0((__int64)v22, v21 + v23, (__int64)&v36, 0, (__m128i)v10);
  *((_QWORD *)&v30 + 1) = v11;
  *(_QWORD *)&v30 = v9;
  v26 = sub_3406EB0(v22, 0xA1u, (__int64)&v36, v33, v31, v25, v30, v24);
  v27 = v36;
  *(_QWORD *)a4 = v26;
  *(_DWORD *)(a4 + 8) = v28;
  if ( v27 )
    sub_B91220((__int64)&v36, v27);
}
