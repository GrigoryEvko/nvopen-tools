// Function: sub_3783390
// Address: 0x3783390
//
void __fastcall sub_3783390(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 *v5; // rax
  __int64 v6; // rsi
  __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rsi
  __m128i v10; // kr00_16
  int v11; // eax
  __int64 v12; // rsi
  unsigned int *v13; // rax
  unsigned __int16 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r11
  int v17; // eax
  unsigned int *v18; // r10
  __int64 v19; // rdx
  __int16 v20; // cx
  __m128i v21; // xmm1
  __int64 v22; // r9
  int v23; // edx
  __int64 v24; // r9
  unsigned __int8 *v25; // rax
  __int64 v26; // rsi
  int v27; // edx
  __int128 *v28; // [rsp+8h] [rbp-128h]
  __int64 v29; // [rsp+10h] [rbp-120h]
  _QWORD *v30; // [rsp+10h] [rbp-120h]
  __int64 v31; // [rsp+30h] [rbp-100h]
  __int64 v32; // [rsp+38h] [rbp-F8h]
  __int64 v35; // [rsp+70h] [rbp-C0h] BYREF
  int v36; // [rsp+78h] [rbp-B8h]
  __int128 v37; // [rsp+80h] [rbp-B0h] BYREF
  __int128 v38; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-90h] BYREF
  int v40; // [rsp+A8h] [rbp-88h]
  __m128i v41; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v42; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v43; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v44; // [rsp+D8h] [rbp-58h]
  __m128i v45; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v46; // [rsp+F0h] [rbp-40h] BYREF

  v5 = *(__int16 **)(a2 + 48);
  v6 = a1[1];
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  LOWORD(v43) = v7;
  v44 = v8;
  sub_33D0340((__int64)&v45, v6, &v43);
  v9 = *(_QWORD *)(a2 + 80);
  v32 = v45.m128i_i64[0];
  v35 = v9;
  v31 = v45.m128i_i64[1];
  v10 = v46;
  if ( v9 )
    sub_B96E90((__int64)&v35, v9, 1);
  v11 = *(_DWORD *)(a2 + 72);
  v12 = *a1;
  DWORD2(v37) = 0;
  v36 = v11;
  v13 = *(unsigned int **)(a2 + 40);
  DWORD2(v38) = 0;
  *(_QWORD *)&v37 = 0;
  *(_QWORD *)&v38 = 0;
  v14 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * v13[2]);
  sub_2FE6CC0((__int64)&v45, v12, *(_QWORD *)(a1[1] + 64), *v14, *((_QWORD *)v14 + 1));
  if ( v45.m128i_i8[0] == 6 )
  {
    sub_375E8D0(
      (__int64)a1,
      **(_QWORD **)(a2 + 40),
      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
      (__int64)&v37,
      (__int64)&v38);
  }
  else
  {
    v15 = *(_QWORD *)(a2 + 80);
    v16 = a1[1];
    v39 = v15;
    if ( v15 )
    {
      v29 = v16;
      sub_B96E90((__int64)&v39, v15, 1);
      v16 = v29;
    }
    v17 = *(_DWORD *)(a2 + 72);
    v18 = *(unsigned int **)(a2 + 40);
    v41.m128i_i64[1] = 0;
    v40 = v17;
    v42.m128i_i64[1] = 0;
    v41.m128i_i16[0] = 0;
    v42.m128i_i16[0] = 0;
    v28 = (__int128 *)v18;
    v30 = (_QWORD *)v16;
    v19 = *(_QWORD *)(*(_QWORD *)v18 + 48LL) + 16LL * v18[2];
    v20 = *(_WORD *)v19;
    v44 = *(_QWORD *)(v19 + 8);
    LOWORD(v43) = v20;
    sub_33D0340((__int64)&v45, v16, &v43);
    v21 = _mm_loadu_si128(&v46);
    v41 = _mm_loadu_si128(&v45);
    v42 = v21;
    sub_3408290((__int64)&v45, v30, v28, (__int64)&v39, (unsigned int *)&v41, (unsigned int *)&v42, v41);
    if ( v39 )
      sub_B91220((__int64)&v39, v39);
    *(_QWORD *)&v37 = v45.m128i_i64[0];
    DWORD2(v37) = v45.m128i_i32[2];
    *(_QWORD *)&v38 = v46.m128i_i64[0];
    DWORD2(v38) = v46.m128i_i32[2];
  }
  *(_QWORD *)a3 = sub_3406EB0(
                    (_QWORD *)a1[1],
                    *(_DWORD *)(a2 + 24),
                    (__int64)&v35,
                    v32,
                    v31,
                    v22,
                    v37,
                    *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  *(_DWORD *)(a3 + 8) = v23;
  v25 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v35,
          v10.m128i_i64[0],
          v10.m128i_i64[1],
          v24,
          v38,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v26 = v35;
  *(_QWORD *)a4 = v25;
  *(_DWORD *)(a4 + 8) = v27;
  if ( v26 )
    sub_B91220((__int64)&v35, v26);
}
