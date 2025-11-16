// Function: sub_38446B0
// Address: 0x38446b0
//
unsigned __int8 *__fastcall sub_38446B0(__int64 a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // r13
  __m128i v6; // xmm0
  __int64 v7; // rsi
  unsigned __int16 *v8; // rax
  unsigned __int64 v9; // rcx
  unsigned int v10; // r14d
  int *v11; // r13
  char v12; // si
  __int64 v13; // r9
  int v14; // ecx
  unsigned int v15; // edi
  __int64 v16; // rax
  int v17; // r10d
  unsigned __int64 v18; // rdx
  __int64 v19; // r13
  __int128 v20; // rax
  unsigned __int8 *v21; // rax
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rsi
  __int64 v26; // r11
  _QWORD *v27; // r14
  unsigned __int16 *v28; // r13
  __int64 v29; // r15
  __int64 v30; // r8
  __int64 v31; // rcx
  unsigned __int8 *v32; // r10
  unsigned __int8 *v33; // r14
  __int64 v35; // rcx
  __int64 v36; // rax
  int v37; // eax
  int v38; // r8d
  __int128 v39; // [rsp-20h] [rbp-A0h]
  unsigned __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  unsigned __int64 v42; // [rsp+10h] [rbp-70h]
  _QWORD *v43; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v46; // [rsp+20h] [rbp-60h]
  __int64 v47; // [rsp+20h] [rbp-60h]
  int v48; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v49; // [rsp+40h] [rbp-40h] BYREF
  int v50; // [rsp+48h] [rbp-38h]

  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4->m128i_i64[0];
  v6 = _mm_loadu_si128(v4);
  v7 = *(_QWORD *)(v4->m128i_i64[0] + 80);
  v8 = (unsigned __int16 *)(*(_QWORD *)(v4->m128i_i64[0] + 48) + 16LL * v4->m128i_u32[2]);
  v9 = *((_QWORD *)v8 + 1);
  v10 = *v8;
  v49 = v7;
  v42 = v9;
  if ( v7 )
    sub_B96E90((__int64)&v49, v7, 1);
  v50 = *(_DWORD *)(v5 + 72);
  v48 = sub_375D5B0(a1, v6.m128i_u64[0], v6.m128i_i64[1]);
  v11 = sub_3805BC0(a1 + 712, &v48);
  sub_37593F0(a1, v11);
  v12 = *(_BYTE *)(a1 + 512) & 1;
  if ( v12 )
  {
    v13 = a1 + 520;
    v14 = 7;
  }
  else
  {
    v35 = *(unsigned int *)(a1 + 528);
    v13 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v35 )
      goto LABEL_16;
    v14 = v35 - 1;
  }
  v15 = v14 & (37 * *v11);
  v16 = v13 + 24LL * v15;
  v17 = *(_DWORD *)v16;
  if ( *v11 == *(_DWORD *)v16 )
    goto LABEL_6;
  v37 = 1;
  while ( v17 != -1 )
  {
    v38 = v37 + 1;
    v15 = v14 & (v37 + v15);
    v16 = v13 + 24LL * v15;
    v17 = *(_DWORD *)v16;
    if ( *v11 == *(_DWORD *)v16 )
      goto LABEL_6;
    v37 = v38;
  }
  if ( v12 )
  {
    v36 = 192;
    goto LABEL_17;
  }
  v35 = *(unsigned int *)(a1 + 528);
LABEL_16:
  v36 = 24 * v35;
LABEL_17:
  v16 = v13 + v36;
LABEL_6:
  v18 = v42;
  v19 = *(unsigned int *)(v16 + 16);
  v40 = *(_QWORD *)(v16 + 8);
  v43 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v20 = sub_33F7D60(v43, v10, v18);
  v21 = sub_3406EB0(
          v43,
          0xDEu,
          (__int64)&v49,
          *(unsigned __int16 *)(*(_QWORD *)(v40 + 48) + 16 * v19),
          *(_QWORD *)(*(_QWORD *)(v40 + 48) + 16 * v19 + 8),
          0xFFFFFFFF00000000LL,
          __PAIR128__(v19 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL, v40),
          v20);
  v24 = v23;
  if ( v49 )
  {
    v46 = v21;
    sub_B91220((__int64)&v49, v49);
    v21 = v46;
  }
  v25 = *(_QWORD *)(a2 + 80);
  v26 = v24;
  v27 = *(_QWORD **)(a1 + 8);
  v28 = (unsigned __int16 *)(*((_QWORD *)v21 + 6) + 16LL * (unsigned int)v24);
  v29 = *(_QWORD *)(a2 + 40);
  v30 = *((_QWORD *)v28 + 1);
  v31 = *v28;
  v32 = v21;
  v49 = v25;
  if ( v25 )
  {
    v41 = v31;
    v45 = v26;
    v47 = v30;
    v44 = v21;
    sub_B96E90((__int64)&v49, v25, 1);
    v31 = v41;
    v32 = v44;
    v26 = v45;
    v30 = v47;
  }
  v50 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v39 + 1) = v26;
  *(_QWORD *)&v39 = v32;
  v33 = sub_3406EB0(v27, 3u, (__int64)&v49, v31, v30, v22, v39, *(_OWORD *)(v29 + 40));
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v33;
}
