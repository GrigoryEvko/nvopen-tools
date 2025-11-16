// Function: sub_3834440
// Address: 0x3834440
//
void __fastcall sub_3834440(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  const __m128i *v7; // rdx
  __int16 *v8; // rax
  unsigned __int16 v9; // di
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int); // r14
  int v11; // eax
  unsigned int v12; // ebx
  __m128i v13; // xmm1
  __m128i v14; // xmm0
  __int64 v15; // rsi
  unsigned __int16 *v16; // rax
  unsigned __int16 *v17; // rax
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // rsi
  int v21; // r13d
  __int64 v22; // rsi
  unsigned __int16 *v23; // rax
  _QWORD *v24; // r8
  __int32 v25; // edx
  __int64 v26; // rdx
  _WORD **v27; // rdi
  int v28; // ecx
  unsigned int v29; // r8d
  _WORD **v30; // rax
  int v31; // r9d
  __int64 v32; // rdx
  __int32 v33; // eax
  __int64 v34; // rax
  int v35; // eax
  int v36; // r10d
  _QWORD *v37; // [rsp+0h] [rbp-110h]
  int *v38; // [rsp+0h] [rbp-110h]
  bool v41; // [rsp+18h] [rbp-F8h]
  unsigned __int16 v42; // [rsp+2Ch] [rbp-E4h]
  char v43; // [rsp+2Fh] [rbp-E1h]
  int *v44; // [rsp+50h] [rbp-C0h] BYREF
  int v45; // [rsp+58h] [rbp-B8h]
  __m128i v46; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v47; // [rsp+70h] [rbp-A0h] BYREF
  int v48; // [rsp+80h] [rbp-90h] BYREF
  __int64 v49; // [rsp+88h] [rbp-88h]
  __int64 v50[4]; // [rsp+90h] [rbp-80h] BYREF
  int *v51; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v52; // [rsp+B8h] [rbp-58h]
  __int64 v53; // [rsp+C0h] [rbp-50h]
  __int64 (__fastcall *v54)(__int64, __int64, unsigned int); // [rsp+C8h] [rbp-48h]
  __int64 v55; // [rsp+D0h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v44 = (int *)v6;
  if ( v6 )
    sub_B96E90((__int64)&v44, v6, 1);
  v7 = *(const __m128i **)(a2 + 40);
  v45 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v8 + 1);
  v11 = *(_DWORD *)(a2 + 24);
  v42 = v9;
  v12 = v9;
  if ( v11 == 226 )
  {
    v41 = 1;
  }
  else
  {
    v41 = v11 == 141;
    if ( v11 > 239 )
    {
      if ( (unsigned int)(v11 - 242) <= 1 )
      {
LABEL_7:
        v13 = _mm_loadu_si128(v7);
        v43 = 1;
        v7 = (const __m128i *)((char *)v7 + 40);
        v46 = v13;
        goto LABEL_8;
      }
    }
    else if ( v11 > 237 || (unsigned int)(v11 - 101) <= 0x2F )
    {
      goto LABEL_7;
    }
  }
  v46.m128i_i64[0] = 0;
  v46.m128i_i32[2] = 0;
  v43 = 0;
LABEL_8:
  v14 = _mm_loadu_si128(v7);
  v15 = *a1;
  v47 = v14;
  v16 = (unsigned __int16 *)(*(_QWORD *)(v14.m128i_i64[0] + 48) + 16LL * v14.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v51, v15, *(_QWORD *)(a1[1] + 64), *v16, *((_QWORD *)v16 + 1));
  if ( (_BYTE)v51 == 8 )
  {
    LODWORD(v51) = sub_375D5B0((__int64)a1, v47.m128i_u64[0], v47.m128i_i64[1]);
    v38 = sub_3805BC0((__int64)(a1 + 123), (int *)&v51);
    sub_37593F0((__int64)a1, v38);
    if ( (a1[64] & 1) != 0 )
    {
      v27 = (_WORD **)(a1 + 65);
      v28 = 7;
    }
    else
    {
      v26 = *((unsigned int *)a1 + 132);
      v27 = (_WORD **)a1[65];
      if ( !(_DWORD)v26 )
        goto LABEL_37;
      v28 = v26 - 1;
    }
    v29 = v28 & (37 * *v38);
    v30 = &v27[3 * v29];
    v31 = *(_DWORD *)v30;
    if ( *v38 == *(_DWORD *)v30 )
    {
LABEL_34:
      v32 = (__int64)v30[1];
      v33 = *((_DWORD *)v30 + 4);
      v47.m128i_i64[0] = v32;
      v47.m128i_i32[2] = v33;
      goto LABEL_9;
    }
    v35 = 1;
    while ( v31 != -1 )
    {
      v36 = v35 + 1;
      v29 = v28 & (v35 + v29);
      v30 = &v27[3 * v29];
      v31 = *(_DWORD *)v30;
      if ( *v38 == *(_DWORD *)v30 )
        goto LABEL_34;
      v35 = v36;
    }
    if ( (a1[64] & 1) != 0 )
    {
      v34 = 24;
      goto LABEL_38;
    }
    v26 = *((unsigned int *)a1 + 132);
LABEL_37:
    v34 = 3 * v26;
LABEL_38:
    v30 = &v27[v34];
    goto LABEL_34;
  }
LABEL_9:
  v17 = (unsigned __int16 *)(*(_QWORD *)(v47.m128i_i64[0] + 48) + 16LL * v47.m128i_u32[2]);
  sub_2FE6CC0((__int64)&v51, *a1, *(_QWORD *)(a1[1] + 64), *v17, *((_QWORD *)v17 + 1));
  if ( (_BYTE)v51 == 9
    || (v18 = *(_QWORD *)(v47.m128i_i64[0] + 48) + 16LL * v47.m128i_u32[2], v19 = *(_WORD *)v18, *(_WORD *)v18 == 10) )
  {
    v24 = (_QWORD *)a1[1];
    v51 = v44;
    if ( v44 )
    {
      v37 = v24;
      sub_B96E90((__int64)&v51, (__int64)v44, 1);
      v24 = v37;
    }
    LODWORD(v52) = v45;
    v47.m128i_i64[0] = (__int64)sub_38136F0(
                                  v47.m128i_i64[0],
                                  v47.m128i_i64[1],
                                  &v46,
                                  v43,
                                  12,
                                  0,
                                  v14,
                                  (__int64)&v51,
                                  v24);
    v47.m128i_i32[2] = v25;
    if ( v51 )
      sub_B91220((__int64)&v51, (__int64)v51);
    v18 = *(_QWORD *)(v47.m128i_i64[0] + 48) + 16LL * v47.m128i_u32[2];
    v19 = *(_WORD *)v18;
  }
  v20 = *(_QWORD *)(v18 + 8);
  LOWORD(v48) = v19;
  v49 = v20;
  if ( v41 )
    v21 = sub_2FE5990(v19, v20, v12);
  else
    v21 = sub_2FE5AE0(v19, v20, v12);
  v51 = 0;
  v52 = 0;
  v22 = *a1;
  v53 = 0;
  v54 = 0;
  LOBYTE(v55) = 4;
  v23 = (unsigned __int16 *)(*(_QWORD *)(v47.m128i_i64[0] + 48) + 16LL * v47.m128i_u32[2]);
  sub_2FE6CC0((__int64)v50, v22, *(_QWORD *)(a1[1] + 64), *v23, *((_QWORD *)v23 + 1));
  if ( LOBYTE(v50[0]) == 3 )
  {
    LOBYTE(v55) = v55 | 0x10;
    v51 = &v48;
    v52 = 1;
    LOWORD(v53) = v42;
    v54 = v10;
  }
  else
  {
    LOBYTE(v55) = v55 | 1;
  }
  sub_3494590(
    (__int64)v50,
    (_WORD *)*a1,
    a1[1],
    v21,
    v12,
    v10,
    (__int64)&v47,
    1u,
    (__int64)v51,
    v52,
    v53,
    (__int64)v54,
    v55,
    (__int64)&v44,
    v46.m128i_i64[0],
    v46.m128i_i64[1]);
  sub_375BC20(a1, v50[0], v50[1], a3, a4, v14);
  if ( v43 )
    sub_3760E70((__int64)a1, a2, 1, v50[2], v50[3]);
  if ( v44 )
    sub_B91220((__int64)&v44, (__int64)v44);
}
