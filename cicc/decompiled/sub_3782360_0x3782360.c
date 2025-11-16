// Function: sub_3782360
// Address: 0x3782360
//
void __fastcall sub_3782360(__int64 *a1, unsigned __int64 a2, int a3, _DWORD *a4, _DWORD *a5, __m128i a6)
{
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int16 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r10
  unsigned int *v21; // r11
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int16 v25; // cx
  __m128i v26; // xmm5
  __int64 v27; // r9
  _QWORD *v28; // rdi
  unsigned int v29; // esi
  __m128i v30; // xmm1
  unsigned __int8 *v31; // rax
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  int v34; // edx
  _QWORD *v35; // rdi
  unsigned int v36; // esi
  __int64 v37; // r9
  int v38; // edx
  unsigned __int64 v39; // r9
  __int64 v40; // rbx
  unsigned __int16 *v41; // rax
  __int64 v42; // r12
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  __int128 v45; // [rsp-20h] [rbp-150h]
  __int128 v46; // [rsp-20h] [rbp-150h]
  __int128 v47; // [rsp-10h] [rbp-140h]
  __int128 v48; // [rsp-10h] [rbp-140h]
  __int128 *v49; // [rsp+8h] [rbp-128h]
  __int64 v50; // [rsp+10h] [rbp-120h]
  _QWORD *v51; // [rsp+10h] [rbp-120h]
  unsigned __int64 v52; // [rsp+10h] [rbp-120h]
  unsigned __int16 v54; // [rsp+1Ch] [rbp-114h]
  unsigned __int64 v55; // [rsp+20h] [rbp-110h]
  __int64 v56; // [rsp+50h] [rbp-E0h] BYREF
  int v57; // [rsp+58h] [rbp-D8h]
  __int64 v58; // [rsp+60h] [rbp-D0h] BYREF
  int v59; // [rsp+68h] [rbp-C8h]
  __m128i v60; // [rsp+70h] [rbp-C0h] BYREF
  __m128i v61; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v62[2]; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v63; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v64; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v65; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v66; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v67; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v68; // [rsp+F0h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 80);
  v56 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v56, v10, 1);
  v11 = a1[1];
  v57 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(a2 + 48);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v67.m128i_i16[0] = v13;
  v67.m128i_i64[1] = v14;
  sub_33D0340((__int64)&v63, v11, v67.m128i_i64);
  v15 = *(_QWORD *)(a2 + 48);
  v16 = a1[1];
  v17 = *(_QWORD *)(v15 + 24);
  LOWORD(v15) = *(_WORD *)(v15 + 16);
  v67.m128i_i64[1] = v17;
  v67.m128i_i16[0] = v15;
  sub_33D0340((__int64)&v65, v16, v67.m128i_i64);
  v18 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  sub_2FE6CC0((__int64)&v67, *a1, *(_QWORD *)(a1[1] + 64), *v18, *((_QWORD *)v18 + 1));
  if ( v67.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a4, (__int64)a5);
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 80);
    v20 = a1[1];
    v58 = v19;
    if ( v19 )
    {
      v50 = v20;
      sub_B96E90((__int64)&v58, v19, 1);
      v20 = v50;
    }
    v21 = *(unsigned int **)(a2 + 40);
    v22 = *(_DWORD *)(a2 + 72);
    v60.m128i_i64[1] = 0;
    v59 = v22;
    v61.m128i_i64[1] = 0;
    v23 = v21[2];
    v61.m128i_i16[0] = 0;
    v60.m128i_i16[0] = 0;
    v49 = (__int128 *)v21;
    v24 = *(_QWORD *)(*(_QWORD *)v21 + 48LL) + 16 * v23;
    v51 = (_QWORD *)v20;
    v25 = *(_WORD *)v24;
    v62[1] = *(_QWORD *)(v24 + 8);
    LOWORD(v62[0]) = v25;
    sub_33D0340((__int64)&v67, v20, v62);
    v26 = _mm_loadu_si128(&v68);
    v60 = _mm_loadu_si128(&v67);
    v61 = v26;
    sub_3408290((__int64)&v67, v51, v49, (__int64)&v58, (unsigned int *)&v60, (unsigned int *)&v61, a6);
    if ( v58 )
      sub_B91220((__int64)&v58, v58);
    *(_QWORD *)a4 = v67.m128i_i64[0];
    a4[2] = v67.m128i_i32[2];
    *(_QWORD *)a5 = v68.m128i_i64[0];
    a5[2] = v68.m128i_i32[2];
  }
  v28 = (_QWORD *)a1[1];
  v29 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v47 + 1) = 1;
  v30 = _mm_loadu_si128(&v65);
  *(_QWORD *)&v47 = a4;
  v67 = _mm_loadu_si128(&v63);
  v68 = v30;
  v31 = sub_3411BE0(v28, v29, (__int64)&v56, (unsigned __int16 *)&v67, 2, v27, v47);
  v32 = _mm_loadu_si128(&v64);
  v33 = _mm_loadu_si128(&v66);
  *(_QWORD *)a4 = v31;
  v67 = v32;
  a4[2] = v34;
  v35 = (_QWORD *)a1[1];
  *((_QWORD *)&v45 + 1) = 1;
  v36 = *(_DWORD *)(a2 + 24);
  *(_QWORD *)&v45 = a5;
  v68 = v33;
  *(_QWORD *)a5 = sub_3411BE0(v35, v36, (__int64)&v56, (unsigned __int16 *)&v67, 2, v37, v45);
  a5[2] = v38;
  *(_DWORD *)(*(_QWORD *)a4 + 28LL) = *(_DWORD *)(a2 + 28);
  *(_DWORD *)(*(_QWORD *)a5 + 28LL) = *(_DWORD *)(a2 + 28);
  v39 = *(_QWORD *)a4;
  v40 = (unsigned int)(1 - a3);
  v55 = *(_QWORD *)a5;
  v41 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v40);
  v52 = v39;
  v42 = *((_QWORD *)v41 + 1);
  v54 = *v41;
  sub_2FE6CC0((__int64)&v67, *a1, *(_QWORD *)(a1[1] + 64), *v41, v42);
  if ( v67.m128i_i8[0] == 6 )
  {
    sub_3760810((__int64)a1, a2, v40, v52, v40, v52, v55, v40);
  }
  else
  {
    *((_QWORD *)&v48 + 1) = v40;
    *(_QWORD *)&v48 = v55;
    *((_QWORD *)&v46 + 1) = v40;
    *(_QWORD *)&v46 = v52;
    v43 = sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v56, v54, v42, v52, v46, v48);
    sub_3760E70((__int64)a1, a2, v40, (unsigned __int64)v43, v44);
  }
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
}
