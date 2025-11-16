// Function: sub_3843E80
// Address: 0x3843e80
//
__m128i *__fastcall sub_3843E80(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v10; // r15
  char v11; // si
  _QWORD *v12; // rdi
  int v13; // ecx
  unsigned int v14; // r9d
  _QWORD *v15; // rax
  int v16; // r8d
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r9
  unsigned __int8 v21; // al
  char v22; // cl
  __int64 v23; // rdi
  __int64 v24; // rsi
  char v25; // r11
  __int64 v26; // rax
  __m128i *v27; // r15
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // eax
  int v33; // r10d
  __int128 v34; // [rsp-40h] [rbp-B0h]
  __int64 v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  int *v37; // [rsp+10h] [rbp-60h]
  char v38; // [rsp+10h] [rbp-60h]
  __m128i v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+20h] [rbp-50h] BYREF
  int v41; // [rsp+28h] [rbp-48h]
  __int64 v42; // [rsp+30h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v40, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v41;
    v10 = v42;
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = v30;
  }
  LODWORD(v40) = sub_375D5B0(
                   (__int64)a1,
                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL),
                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL));
  v37 = sub_3805BC0((__int64)(a1 + 89), (int *)&v40);
  sub_37593F0((__int64)a1, v37);
  v11 = a1[64] & 1;
  if ( v11 )
  {
    v12 = a1 + 65;
    v13 = 7;
  }
  else
  {
    v29 = *((unsigned int *)a1 + 132);
    v12 = (_QWORD *)a1[65];
    if ( !(_DWORD)v29 )
      goto LABEL_17;
    v13 = v29 - 1;
  }
  v14 = v13 & (37 * *v37);
  v15 = &v12[3 * v14];
  v16 = *(_DWORD *)v15;
  if ( *v37 == *(_DWORD *)v15 )
    goto LABEL_6;
  v32 = 1;
  while ( v16 != -1 )
  {
    v33 = v32 + 1;
    v14 = v13 & (v32 + v14);
    v15 = &v12[3 * v14];
    v16 = *(_DWORD *)v15;
    if ( *v37 == *(_DWORD *)v15 )
      goto LABEL_6;
    v32 = v33;
  }
  if ( v11 )
  {
    v31 = 24;
    goto LABEL_18;
  }
  v29 = *((unsigned int *)a1 + 132);
LABEL_17:
  v31 = 3 * v29;
LABEL_18:
  v15 = &v12[v31];
LABEL_6:
  v17 = v15[1];
  v18 = *(_QWORD *)(a2 + 80);
  v19 = *((unsigned int *)v15 + 4);
  v40 = v18;
  v20 = v19;
  v21 = *(_BYTE *)(a2 + 33);
  v22 = (v21 >> 2) & 3;
  if ( !v22 )
    v22 = 1;
  if ( v18 )
  {
    v35 = v17;
    v36 = v20;
    v38 = v22;
    sub_B96E90((__int64)&v40, v18, 1);
    v21 = *(_BYTE *)(a2 + 33);
    v17 = v35;
    v20 = v36;
    v22 = v38;
  }
  v23 = *(_QWORD *)(a2 + 104);
  v24 = *(unsigned __int16 *)(a2 + 96);
  v41 = *(_DWORD *)(a2 + 72);
  v25 = (v21 & 0x10) != 0;
  v26 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v34 + 1) = v20;
  *(_QWORD *)&v34 = v17;
  v39 = _mm_loadu_si128((const __m128i *)v26);
  v27 = sub_33E8F60(
          (__int64 *)a1[1],
          v9,
          v10,
          (__int64)&v40,
          v39.m128i_u64[0],
          v39.m128i_i64[1],
          *(_QWORD *)(v26 + 40),
          *(_QWORD *)(v26 + 48),
          *(_OWORD *)(v26 + 80),
          *(_OWORD *)(v26 + 120),
          v34,
          v24,
          v23,
          *(const __m128i **)(a2 + 112),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          v22,
          v25);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v27, 1);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v27;
}
