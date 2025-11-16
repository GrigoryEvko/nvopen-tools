// Function: sub_379EE60
// Address: 0x379ee60
//
__int64 __fastcall sub_379EE60(__int64 *a1, __int64 a2)
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
  unsigned __int64 v24; // rax
  __int64 v25; // r12
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rax
  unsigned int v31; // edx
  __int64 v32; // rax
  unsigned int v33; // edx
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int128 v38; // [rsp-20h] [rbp-E0h]
  __int64 v39; // [rsp+0h] [rbp-C0h]
  char v40; // [rsp+Fh] [rbp-B1h]
  unsigned __int64 v41; // [rsp+18h] [rbp-A8h]
  __int64 v42; // [rsp+40h] [rbp-80h] BYREF
  int v43; // [rsp+48h] [rbp-78h]
  unsigned int v44; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+58h] [rbp-68h]
  unsigned __int16 v46; // [rsp+60h] [rbp-60h] BYREF
  __int64 v47; // [rsp+68h] [rbp-58h]
  __int64 v48; // [rsp+70h] [rbp-50h] BYREF
  __int16 v49; // [rsp+78h] [rbp-48h]
  __int64 v50; // [rsp+80h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v42 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v42, v3, 1);
  v4 = *a1;
  v43 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v48, v4, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v10) = v49;
    LOWORD(v44) = v49;
    v45 = v50;
  }
  else
  {
    v10 = v5(v4, *(_QWORD *)(v9 + 64), v7, v8);
    v44 = v10;
    v45 = v37;
  }
  if ( (_WORD)v10 )
  {
    v40 = (unsigned __int16)(v10 - 176) <= 0x34u;
    v11 = word_4456340[(unsigned __int16)v10 - 1];
  }
  else
  {
    v32 = sub_3007240((__int64)&v44);
    v11 = v32;
    v40 = BYTE4(v32);
  }
  v12 = *(const __m128i **)(a2 + 40);
  v13 = *a1;
  v14 = _mm_loadu_si128(v12);
  v15 = v12->m128i_u32[2];
  v16 = *(_QWORD *)(v12->m128i_i64[0] + 48) + 16 * v15;
  v39 = v12->m128i_i64[0];
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v41 = v14.m128i_u64[1];
  v19 = a1[1];
  v46 = v17;
  v20 = *(_QWORD *)(v19 + 64);
  v47 = v18;
  sub_2FE6CC0((__int64)&v48, v13, v20, v17, v18);
  if ( (_BYTE)v48 == 7 )
  {
    v39 = sub_379AB60((__int64)a1, v14.m128i_u64[0], v14.m128i_i64[1]);
    v15 = v33;
    v34 = v33 | v14.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v35 = *(_QWORD *)(v39 + 48) + 16LL * v33;
    v17 = *(_WORD *)v35;
    v36 = *(_QWORD *)(v35 + 8);
    v41 = v34;
    v46 = v17;
    v47 = v36;
  }
  if ( v17 )
  {
    v23 = word_4456340;
    LOBYTE(v24) = (unsigned __int16)(v17 - 176) <= 0x34u;
    if ( v11 != word_4456340[v17 - 1] )
    {
LABEL_11:
      v25 = (__int64)sub_3412A00((_QWORD *)a1[1], a2, v11, (__int64)v23, v21, v22, v14);
      goto LABEL_12;
    }
  }
  else
  {
    v27 = sub_3007240((__int64)&v46);
    v24 = HIDWORD(v27);
    if ( v11 != (_DWORD)v27 )
      goto LABEL_11;
  }
  if ( v40 != (_BYTE)v24 )
    goto LABEL_11;
  if ( *(_DWORD *)(a2 + 64) == 1 )
  {
    v25 = (__int64)sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v42, v44, v45, v22, v14);
  }
  else
  {
    if ( !(_WORD)v44 )
      v48 = sub_3007240((__int64)&v44);
    v28 = *(_QWORD *)(a2 + 40);
    v29 = *(_QWORD *)(v28 + 48);
    v30 = sub_379AB60((__int64)a1, *(_QWORD *)(v28 + 40), v29);
    *((_QWORD *)&v38 + 1) = v31 | v29 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v38 = v30;
    v25 = sub_340F900(
            (_QWORD *)a1[1],
            *(_DWORD *)(a2 + 24),
            (__int64)&v42,
            v44,
            v45,
            *(_QWORD *)(a2 + 40),
            __PAIR128__(v15 | v41 & 0xFFFFFFFF00000000LL, v39),
            v38,
            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
  }
LABEL_12:
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  return v25;
}
