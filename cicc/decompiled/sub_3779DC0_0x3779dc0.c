// Function: sub_3779DC0
// Address: 0x3779dc0
//
void __fastcall sub_3779DC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rax
  unsigned __int16 v14; // r15
  __int64 v15; // r8
  unsigned __int16 *v16; // rax
  unsigned __int8 *v17; // rax
  int v18; // edx
  __int64 v19; // rdx
  unsigned __int16 *v20; // rax
  __int64 v21; // r9
  int v22; // edx
  bool v23; // al
  __int64 v24; // rcx
  __int64 v25; // r10
  __int64 v26; // rsi
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // rax
  __int16 v30; // dx
  __int64 v31; // rax
  __m128i v32; // xmm2
  __int64 v33; // r9
  unsigned __int16 *v34; // rax
  unsigned __int8 *v35; // rax
  int v36; // edx
  __int64 v37; // rdx
  unsigned __int16 *v38; // rax
  __int64 v39; // r9
  int v40; // edx
  __int64 v41; // [rsp+0h] [rbp-160h]
  __int64 v42; // [rsp+8h] [rbp-158h]
  _QWORD *v43; // [rsp+8h] [rbp-158h]
  __int64 v44; // [rsp+10h] [rbp-150h]
  int v46; // [rsp+38h] [rbp-128h]
  int v47; // [rsp+58h] [rbp-108h]
  __int128 v48; // [rsp+60h] [rbp-100h] BYREF
  __int128 v49; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v50; // [rsp+80h] [rbp-E0h] BYREF
  int v51; // [rsp+88h] [rbp-D8h]
  __int128 v52; // [rsp+90h] [rbp-D0h] BYREF
  __int128 v53; // [rsp+A0h] [rbp-C0h] BYREF
  __m128i v54; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned __int16 v55; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v56; // [rsp+C8h] [rbp-98h]
  __int64 v57; // [rsp+D0h] [rbp-90h] BYREF
  int v58; // [rsp+D8h] [rbp-88h]
  __m128i v59; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v60; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v61[2]; // [rsp+100h] [rbp-60h] BYREF
  __m128i v62; // [rsp+110h] [rbp-50h] BYREF
  __m128i v63; // [rsp+120h] [rbp-40h] BYREF

  v6 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v48) = 0;
  DWORD2(v49) = 0;
  *(_QWORD *)&v48 = 0;
  v7 = v6[1];
  *(_QWORD *)&v49 = 0;
  sub_375E8D0(a1, *v6, v7, (__int64)&v48, (__int64)&v49);
  v9 = *(_QWORD *)(a2 + 80);
  v50 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v50, v9, 1);
  DWORD2(v52) = 0;
  v10 = *(_DWORD *)(a2 + 72);
  DWORD2(v53) = 0;
  v51 = v10;
  v11 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)&v52 = 0;
  *(_QWORD *)&v53 = 0;
  v12 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v54 = v12;
  v13 = *(_QWORD *)(v12.m128i_i64[0] + 48) + 16LL * v12.m128i_u32[2];
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v55 = v14;
  v56 = v15;
  if ( v14 )
  {
    if ( (unsigned __int16)(v14 - 17) > 0xD3u )
    {
LABEL_5:
      v16 = (unsigned __int16 *)(*(_QWORD *)(v48 + 48) + 16LL * DWORD2(v48));
      v17 = sub_3406EB0(
              *(_QWORD **)(a1 + 8),
              *(_DWORD *)(a2 + 24),
              (__int64)&v50,
              *v16,
              *((_QWORD *)v16 + 1),
              v8,
              v48,
              *(_OWORD *)&v54);
      v46 = v18;
      v19 = v49;
      *(_QWORD *)a3 = v17;
      *(_DWORD *)(a3 + 8) = v46;
      v20 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * DWORD2(v49));
      *(_QWORD *)a4 = sub_3406EB0(
                        *(_QWORD **)(a1 + 8),
                        *(_DWORD *)(a2 + 24),
                        (__int64)&v50,
                        *v20,
                        *((_QWORD *)v20 + 1),
                        v21,
                        v49,
                        *(_OWORD *)&v54);
      *(_DWORD *)(a4 + 8) = v22;
      goto LABEL_6;
    }
  }
  else
  {
    v44 = v15;
    v23 = sub_30070B0((__int64)&v55);
    v15 = v44;
    if ( !v23 )
      goto LABEL_5;
  }
  sub_2FE6CC0((__int64)&v62, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v14, v15);
  if ( v62.m128i_i8[0] == 6 )
  {
    sub_375E8D0(a1, v54.m128i_u64[0], v54.m128i_i64[1], (__int64)&v52, (__int64)&v53);
  }
  else
  {
    v24 = v54.m128i_i64[0];
    v25 = *(_QWORD *)(a1 + 8);
    v26 = *(_QWORD *)(v54.m128i_i64[0] + 80);
    v57 = v26;
    if ( v26 )
    {
      v41 = v54.m128i_i64[0];
      v42 = v25;
      sub_B96E90((__int64)&v57, v26, 1);
      v27 = v54.m128i_i64[0];
      v25 = v42;
      v24 = v41;
    }
    else
    {
      v27 = v54.m128i_i64[0];
    }
    v28 = *(_DWORD *)(v24 + 72);
    v59.m128i_i64[1] = 0;
    v60.m128i_i16[0] = 0;
    v58 = v28;
    v59.m128i_i16[0] = 0;
    v60.m128i_i64[1] = 0;
    v29 = *(_QWORD *)(v27 + 48) + 16LL * v54.m128i_u32[2];
    v43 = (_QWORD *)v25;
    v30 = *(_WORD *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    LOWORD(v61[0]) = v30;
    v61[1] = v31;
    sub_33D0340((__int64)&v62, v25, v61);
    v32 = _mm_loadu_si128(&v63);
    v59 = _mm_loadu_si128(&v62);
    v60 = v32;
    sub_3408290(
      (__int64)&v62,
      v43,
      (__int128 *)v54.m128i_i8,
      (__int64)&v57,
      (unsigned int *)&v59,
      (unsigned int *)&v60,
      v12);
    *(_QWORD *)&v52 = v62.m128i_i64[0];
    DWORD2(v52) = v62.m128i_i32[2];
    *(_QWORD *)&v53 = v63.m128i_i64[0];
    DWORD2(v53) = v63.m128i_i32[2];
    if ( v57 )
      sub_B91220((__int64)&v57, v57);
  }
  v34 = (unsigned __int16 *)(*(_QWORD *)(v48 + 48) + 16LL * DWORD2(v48));
  v35 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          *(_DWORD *)(a2 + 24),
          (__int64)&v50,
          *v34,
          *((_QWORD *)v34 + 1),
          v33,
          v48,
          v52);
  v47 = v36;
  v37 = v49;
  *(_QWORD *)a3 = v35;
  *(_DWORD *)(a3 + 8) = v47;
  v38 = (unsigned __int16 *)(*(_QWORD *)(v37 + 48) + 16LL * DWORD2(v49));
  *(_QWORD *)a4 = sub_3406EB0(
                    *(_QWORD **)(a1 + 8),
                    *(_DWORD *)(a2 + 24),
                    (__int64)&v50,
                    *v38,
                    *((_QWORD *)v38 + 1),
                    v39,
                    v49,
                    v53);
  *(_DWORD *)(a4 + 8) = v40;
LABEL_6:
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
}
