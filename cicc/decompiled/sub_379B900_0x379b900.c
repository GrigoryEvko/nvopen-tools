// Function: sub_379B900
// Address: 0x379b900
//
__int64 __fastcall sub_379B900(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r11
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r12
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  __m128i v10; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // r13
  __int128 v14; // rax
  __int128 v15; // kr00_16
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rdx
  __m128i v21; // xmm0
  const __m128i *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r12
  __int64 v26; // rdx
  __int128 v27; // [rsp-20h] [rbp-110h]
  __int128 v28; // [rsp-10h] [rbp-100h]
  __int128 v29; // [rsp+0h] [rbp-F0h]
  __m128i v30; // [rsp+10h] [rbp-E0h] BYREF
  __int64 *v31; // [rsp+28h] [rbp-C8h]
  __int64 v32; // [rsp+30h] [rbp-C0h]
  __int64 v33; // [rsp+38h] [rbp-B8h]
  __int64 v34; // [rsp+48h] [rbp-A8h]
  __int64 v35; // [rsp+50h] [rbp-A0h] BYREF
  int v36; // [rsp+58h] [rbp-98h]
  unsigned int v37; // [rsp+60h] [rbp-90h] BYREF
  __int64 v38; // [rsp+68h] [rbp-88h]
  __m128i v39; // [rsp+70h] [rbp-80h] BYREF
  __int64 v40; // [rsp+80h] [rbp-70h]
  __int64 v41; // [rsp+88h] [rbp-68h]
  __int128 v42; // [rsp+90h] [rbp-60h]
  __int64 v43; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v44; // [rsp+A8h] [rbp-48h]
  __m128i v45; // [rsp+B0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 80);
  v31 = &v35;
  v35 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v35, v3, 1);
  v4 = *a1;
  v36 = *(_DWORD *)(a2 + 72);
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v4 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v39, v4, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v37) = v39.m128i_i16[4];
    v38 = v40;
  }
  else
  {
    v37 = v5(v4, *(_QWORD *)(v9 + 64), v7, v8);
    v38 = v26;
  }
  v10.m128i_i64[0] = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v30 = v10;
  v11 = sub_379AB60((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v13 = v12;
  *(_QWORD *)&v14 = sub_379AB60(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v15 = v14;
  if ( *(_DWORD *)(a2 + 64) == 3 )
  {
    *((_QWORD *)&v27 + 1) = v13;
    *(_QWORD *)&v27 = v11;
    v24 = sub_340F900(
            (_QWORD *)a1[1],
            *(_DWORD *)(a2 + 24),
            (__int64)v31,
            v37,
            v38,
            *((__int64 *)&v14 + 1),
            *(_OWORD *)&v30,
            v27,
            v14);
  }
  else
  {
    if ( !(_WORD)v37 )
    {
      v29 = v14;
      v34 = sub_3007240((__int64)&v37);
      v15 = v29;
    }
    v16 = *(_QWORD *)(a2 + 40);
    v17 = *(_QWORD *)(v16 + 128);
    v18 = sub_379AB60((__int64)a1, *(_QWORD *)(v16 + 120), v17);
    v40 = v11;
    v19 = (_QWORD *)a1[1];
    v33 = v20;
    v43 = v18;
    v32 = v18;
    v21 = _mm_load_si128(&v30);
    v22 = *(const __m128i **)(a2 + 40);
    v41 = v13;
    v42 = v15;
    v23 = *(unsigned int *)(a2 + 24);
    v44 = (unsigned int)v20 | v17 & 0xFFFFFFFF00000000LL;
    v39 = v21;
    *((_QWORD *)&v28 + 1) = 5;
    *(_QWORD *)&v28 = &v39;
    v45 = _mm_loadu_si128(v22 + 10);
    v24 = (__int64)sub_33FC220(v19, v23, (__int64)v31, v37, v38, *((__int64 *)&v15 + 1), v28);
  }
  if ( v35 )
    sub_B91220((__int64)v31, v35);
  return v24;
}
