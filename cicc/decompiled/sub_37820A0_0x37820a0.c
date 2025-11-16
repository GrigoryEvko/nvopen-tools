// Function: sub_37820A0
// Address: 0x37820a0
//
void __fastcall sub_37820A0(__int64 *a1, __int64 a2, unsigned __int64 *a3, unsigned __int64 *a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  unsigned __int16 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r10
  int v14; // eax
  unsigned int *v15; // r11
  __int64 v16; // rdx
  __int16 v17; // cx
  __m128i v18; // xmm1
  int v19; // r15d
  int v20; // ebx
  _QWORD *v21; // rax
  int v22; // edx
  __int64 v23; // rcx
  unsigned int v24; // edx
  _QWORD *v25; // rax
  __int64 v26; // rsi
  int v27; // edx
  __int128 *v28; // [rsp+0h] [rbp-100h]
  __int64 v29; // [rsp+8h] [rbp-F8h]
  _QWORD *v30; // [rsp+8h] [rbp-F8h]
  int v32; // [rsp+38h] [rbp-C8h]
  __int64 v33; // [rsp+40h] [rbp-C0h] BYREF
  int v34; // [rsp+48h] [rbp-B8h]
  __int64 v35; // [rsp+50h] [rbp-B0h] BYREF
  int v36; // [rsp+58h] [rbp-A8h]
  __m128i v37; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v38; // [rsp+70h] [rbp-90h] BYREF
  __int64 v39[2]; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+90h] [rbp-70h] BYREF
  __int64 v41; // [rsp+98h] [rbp-68h]
  unsigned int v42; // [rsp+A0h] [rbp-60h]
  __int64 v43; // [rsp+A8h] [rbp-58h]
  __m128i v44; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v45; // [rsp+C0h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v33 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v33, v6, 1);
  v7 = a1[1];
  v34 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a2 + 48);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v44.m128i_i16[0] = v9;
  v44.m128i_i64[1] = v10;
  sub_33D0340((__int64)&v40, v7, v44.m128i_i64);
  v11 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  sub_2FE6CC0((__int64)&v44, *a1, *(_QWORD *)(a1[1] + 64), *v11, *((_QWORD *)v11 + 1));
  if ( v44.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, (__int64)a4);
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 80);
    v13 = a1[1];
    v35 = v12;
    if ( v12 )
    {
      v29 = v13;
      sub_B96E90((__int64)&v35, v12, 1);
      v13 = v29;
    }
    v14 = *(_DWORD *)(a2 + 72);
    v15 = *(unsigned int **)(a2 + 40);
    v37.m128i_i64[1] = 0;
    v36 = v14;
    v38.m128i_i64[1] = 0;
    v37.m128i_i16[0] = 0;
    v38.m128i_i16[0] = 0;
    v28 = (__int128 *)v15;
    v30 = (_QWORD *)v13;
    v16 = *(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * v15[2];
    v17 = *(_WORD *)v16;
    v39[1] = *(_QWORD *)(v16 + 8);
    LOWORD(v39[0]) = v17;
    sub_33D0340((__int64)&v44, v13, v39);
    v18 = _mm_loadu_si128(&v45);
    v37 = _mm_loadu_si128(&v44);
    v38 = v18;
    sub_3408290((__int64)&v44, v30, v28, (__int64)&v35, (unsigned int *)&v37, (unsigned int *)&v38, v37);
    if ( v35 )
      sub_B91220((__int64)&v35, v35);
    *a3 = v44.m128i_i64[0];
    *((_DWORD *)a3 + 2) = v44.m128i_i32[2];
    *a4 = v45.m128i_i64[0];
    *((_DWORD *)a4 + 2) = v45.m128i_i32[2];
  }
  v19 = *(_DWORD *)(a2 + 96);
  v20 = *(_DWORD *)(a2 + 100);
  v21 = sub_33F2D30((_QWORD *)a1[1], (__int64)&v33, v40, v41, *a3, a3[1], v19, v20);
  v32 = v22;
  v23 = v43;
  *a3 = (unsigned __int64)v21;
  v24 = v42;
  *((_DWORD *)a3 + 2) = v32;
  v25 = sub_33F2D30((_QWORD *)a1[1], (__int64)&v33, v24, v23, *a4, a4[1], v19, v20);
  v26 = v33;
  *a4 = (unsigned __int64)v25;
  *((_DWORD *)a4 + 2) = v27;
  if ( v26 )
    sub_B91220((__int64)&v33, v26);
}
