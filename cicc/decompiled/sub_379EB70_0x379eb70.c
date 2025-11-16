// Function: sub_379EB70
// Address: 0x379eb70
//
unsigned __int8 *__fastcall sub_379EB70(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r12
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  unsigned int v10; // eax
  unsigned int v11; // r12d
  const __m128i *v12; // rax
  __int64 v13; // rsi
  __m128i v14; // xmm0
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned __int16 v17; // r15
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int16 *v23; // rcx
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rax
  unsigned __int8 *v26; // r12
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // edx
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // [rsp+0h] [rbp-B0h]
  char v37; // [rsp+Fh] [rbp-A1h]
  unsigned __int64 v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+30h] [rbp-80h] BYREF
  int v40; // [rsp+38h] [rbp-78h]
  unsigned int v41; // [rsp+40h] [rbp-70h] BYREF
  __int64 v42; // [rsp+48h] [rbp-68h]
  unsigned __int16 v43; // [rsp+50h] [rbp-60h] BYREF
  __int64 v44; // [rsp+58h] [rbp-58h]
  unsigned __int64 v45; // [rsp+60h] [rbp-50h] BYREF
  __int16 v46; // [rsp+68h] [rbp-48h]
  __int64 v47; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v39 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v39, v3, 1);
  v4 = *a1;
  v40 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v45, v4, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v10) = v46;
    LOWORD(v41) = v46;
    v42 = v47;
  }
  else
  {
    v10 = v5(v4, *(_QWORD *)(v9 + 64), v7, v8);
    v41 = v10;
    v42 = v35;
  }
  if ( (_WORD)v10 )
  {
    v37 = (unsigned __int16)(v10 - 176) <= 0x34u;
    v11 = word_4456340[(unsigned __int16)v10 - 1];
  }
  else
  {
    v30 = sub_3007240((__int64)&v41);
    v11 = v30;
    v37 = BYTE4(v30);
  }
  v12 = *(const __m128i **)(a2 + 40);
  v13 = *a1;
  v14 = _mm_loadu_si128(v12);
  v15 = v12->m128i_u32[2];
  v16 = *(_QWORD *)(v12->m128i_i64[0] + 48) + 16 * v15;
  v36 = v12->m128i_i64[0];
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v38 = v14.m128i_u64[1];
  v19 = a1[1];
  v43 = v17;
  v20 = *(_QWORD *)(v19 + 64);
  v44 = v18;
  sub_2FE6CC0((__int64)&v45, v13, v20, v17, v18);
  if ( (_BYTE)v45 == 7 )
  {
    v36 = sub_379AB60((__int64)a1, v14.m128i_u64[0], v14.m128i_i64[1]);
    v15 = v31;
    v32 = v31 | v14.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v33 = *(_QWORD *)(v36 + 48) + 16LL * v31;
    v17 = *(_WORD *)v33;
    v34 = *(_QWORD *)(v33 + 8);
    v38 = v32;
    v43 = v17;
    v44 = v34;
  }
  if ( v17 )
  {
    v23 = word_4456340;
    v24 = (_QWORD *)a1[1];
    LOBYTE(v25) = (unsigned __int16)(v17 - 176) <= 0x34u;
    if ( v11 != word_4456340[v17 - 1] )
    {
LABEL_11:
      v26 = sub_3412A00(v24, a2, v11, (__int64)v23, v21, v22, v14);
      goto LABEL_12;
    }
  }
  else
  {
    v28 = sub_3007240((__int64)&v43);
    v24 = (_QWORD *)a1[1];
    v29 = v28;
    v25 = HIDWORD(v28);
    v45 = v29;
    if ( v11 != (_DWORD)v29 )
      goto LABEL_11;
  }
  if ( v37 != (_BYTE)v25 )
    goto LABEL_11;
  v26 = sub_3406EB0(
          v24,
          *(_DWORD *)(a2 + 24),
          (__int64)&v39,
          v41,
          v42,
          v22,
          __PAIR128__(v15 | v38 & 0xFFFFFFFF00000000LL, v36),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
LABEL_12:
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v26;
}
