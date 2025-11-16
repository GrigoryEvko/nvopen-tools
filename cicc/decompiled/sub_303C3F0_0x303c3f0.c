// Function: sub_303C3F0
// Address: 0x303c3f0
//
__int64 __fastcall sub_303C3F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rax
  __int64 *v12; // r15
  __int64 v13; // rbx
  __m128i v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // r9d
  __int64 v18; // r14
  __int64 v19; // rsi
  __int128 v20; // rax
  int v21; // r9d
  __int64 v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rdx
  int v25; // r9d
  __int64 v26; // r14
  __int128 v28; // [rsp-10h] [rbp-110h]
  int v29; // [rsp+0h] [rbp-100h]
  __int64 v30; // [rsp+8h] [rbp-F8h]
  int v31; // [rsp+14h] [rbp-ECh]
  __int64 v32; // [rsp+18h] [rbp-E8h]
  __m128i v33; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v34; // [rsp+30h] [rbp-D0h]
  __int64 v35; // [rsp+38h] [rbp-C8h]
  __int128 v36; // [rsp+40h] [rbp-C0h]
  __int64 v37; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v38; // [rsp+58h] [rbp-A8h]
  __int64 v39; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v40; // [rsp+68h] [rbp-98h]
  __int64 v41; // [rsp+70h] [rbp-90h] BYREF
  int v42; // [rsp+78h] [rbp-88h]
  __int64 v43; // [rsp+80h] [rbp-80h] BYREF
  int v44; // [rsp+88h] [rbp-78h]
  __int64 v45; // [rsp+90h] [rbp-70h]
  __int64 v46; // [rsp+98h] [rbp-68h]
  __int64 v47; // [rsp+A0h] [rbp-60h]
  int v48; // [rsp+A8h] [rbp-58h]
  __m128i v49; // [rsp+B0h] [rbp-50h]
  __int64 v50; // [rsp+C0h] [rbp-40h]
  int v51; // [rsp+C8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v41 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v41, v6, 1);
  v42 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v30 = *(_QWORD *)(v7 + 80);
  v8 = *(_DWORD *)(v7 + 88);
  v9 = *(_QWORD *)(v7 + 40);
  v36 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v31 = v8;
  v10 = *(_DWORD *)(v9 + 96);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 64LL) + 8LL) + 32LL * v10;
  v12 = *(__int64 **)v11;
  v13 = *(_QWORD *)(v11 + 8) - *(_QWORD *)v11;
  v14.m128i_i64[0] = sub_3400BD0(a4, v10, (unsigned int)&v41, 7, 0, 0, 0);
  v33 = v14;
  v15 = sub_33E5110(a4, 1, 0, 262, 0);
  v29 = v16;
  v35 = v16;
  v32 = v15;
  v18 = sub_3411F20(a4, 544, (unsigned int)&v41, v15, v16, v17, v36, *(_OWORD *)&v33);
  *(_QWORD *)&v36 = (char *)v12 + v13 - 8;
  if ( v12 != (__int64 *)v36 )
  {
    v34 = v32;
    do
    {
      v19 = *v12;
      v37 = v18;
      ++v12;
      v38 = v38 & 0xFFFFFFFF00000000LL | 1;
      *(_QWORD *)&v20 = sub_33EEAD0(a4, v19);
      v39 = v18;
      v40 &= 0xFFFFFFFF00000000LL;
      v18 = sub_3412970(a4, 545, (unsigned int)&v41, v34, v35, v21, __PAIR128__(v40, v18), v20, __PAIR128__(v38, v18));
    }
    while ( (__int64 *)v36 != v12 );
  }
  v43 = v18;
  v44 = 0;
  v22 = sub_33EEAD0(a4, *(_QWORD *)v36);
  v23 = _mm_load_si128(&v33);
  v45 = v22;
  v46 = v24;
  v47 = v30;
  *((_QWORD *)&v28 + 1) = 5;
  v48 = v31;
  *(_QWORD *)&v28 = &v43;
  v50 = v18;
  v51 = 1;
  v49 = v23;
  v26 = sub_3411630(a4, 546, (unsigned int)&v41, v32, v29, v25, v28);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v26;
}
