// Function: sub_3846760
// Address: 0x3846760
//
void __fastcall sub_3846760(__int64 *a1, unsigned __int64 a2, __m128i *a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rdx
  __int16 *v11; // rax
  __int64 v12; // r10
  unsigned __int16 v13; // si
  __int64 v14; // r8
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 *v18; // rdi
  __int64 v19; // r14
  __int64 v20; // r9
  __m128i v21; // xmm1
  __int64 v22; // rax
  __int32 v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  _QWORD *v27; // rdi
  __int64 *v28; // rdi
  unsigned int v29; // edx
  __int64 v30; // rax
  __int16 v31; // r8
  __int16 v32; // dx
  unsigned __int64 v33; // r9
  int v34; // esi
  __int64 v35; // rcx
  __m128i *v36; // rax
  int v37; // edx
  unsigned __int8 *v38; // r14
  unsigned int v39; // edx
  __m128i v40; // xmm0
  __int64 v41; // rcx
  char v42; // r10
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int128 v45; // [rsp-20h] [rbp-150h]
  __int128 v46; // [rsp-10h] [rbp-140h]
  unsigned __int16 v47; // [rsp+Eh] [rbp-122h]
  __int64 v48; // [rsp+10h] [rbp-120h]
  __int64 v49; // [rsp+20h] [rbp-110h]
  unsigned __int64 v51; // [rsp+30h] [rbp-100h]
  unsigned __int64 v52; // [rsp+38h] [rbp-F8h]
  unsigned __int8 *v53; // [rsp+70h] [rbp-C0h]
  __int64 v54; // [rsp+90h] [rbp-A0h] BYREF
  int v55; // [rsp+98h] [rbp-98h]
  unsigned int v56; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-88h]
  __int64 v58; // [rsp+B0h] [rbp-80h]
  __int64 v59; // [rsp+B8h] [rbp-78h]
  __int128 v60; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v61; // [rsp+D0h] [rbp-60h]
  __m128i v62; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v63; // [rsp+F0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 80);
  v54 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v54, v8, 1);
  v9 = *a1;
  v10 = a1[1];
  v55 = *(_DWORD *)(a2 + 72);
  v11 = *(__int16 **)(a2 + 48);
  v12 = *(_QWORD *)(v10 + 64);
  v13 = *v11;
  v14 = *((_QWORD *)v11 + 1);
  v47 = *v11;
  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v9 + 592LL);
  if ( v15 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v62, v9, v12, v13, v14);
    LOWORD(v56) = v62.m128i_i16[4];
    v57 = v63.m128i_i64[0];
  }
  else
  {
    v56 = v15(v9, v12, v47, v14);
    v57 = v43;
  }
  v16 = *(_QWORD *)(a2 + 40);
  HIBYTE(v17) = 1;
  v18 = (__int64 *)a1[1];
  v19 = *(_QWORD *)v16;
  v20 = *(_QWORD *)(v16 + 8);
  v21 = _mm_loadu_si128((const __m128i *)(v16 + 40));
  v22 = *(_QWORD *)(a2 + 112);
  v49 = v20;
  v62 = _mm_loadu_si128((const __m128i *)(v22 + 40));
  v63 = _mm_loadu_si128((const __m128i *)(v22 + 56));
  LOBYTE(v17) = *(_BYTE *)(v22 + 34);
  a3->m128i_i64[0] = (__int64)sub_33F1F00(
                                v18,
                                v56,
                                v57,
                                (__int64)&v54,
                                v19,
                                v20,
                                v21.m128i_i64[0],
                                v21.m128i_i64[1],
                                *(_OWORD *)v22,
                                *(_QWORD *)(v22 + 16),
                                v17,
                                *(_WORD *)(v22 + 32),
                                (__int64)&v62,
                                0);
  a3->m128i_i32[2] = v23;
  if ( (_WORD)v56 )
  {
    if ( (_WORD)v56 == 1 || (unsigned __int16)(v56 - 504) <= 7u )
      BUG();
    v25 = 16LL * ((unsigned __int16)v56 - 1);
    v24 = *(_QWORD *)&byte_444C4A0[v25];
    LOBYTE(v25) = byte_444C4A0[v25 + 8];
  }
  else
  {
    v24 = sub_3007260((__int64)&v56);
    v58 = v24;
    v59 = v25;
  }
  BYTE8(v60) = v25;
  *(_QWORD *)&v60 = v24;
  v26 = sub_CA1930(&v60);
  BYTE8(v60) = 0;
  v27 = (_QWORD *)a1[1];
  *(_QWORD *)&v60 = (unsigned int)(v26 >> 3);
  v48 = v60;
  v53 = sub_3409320(v27, v21.m128i_i64[0], v21.m128i_i64[1], v60, 0, (__int64)&v54, a5, 1);
  v28 = (__int64 *)a1[1];
  v52 = v29 | v21.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v30 = *(_QWORD *)(a2 + 112);
  LOBYTE(v32) = *(_BYTE *)(v30 + 34);
  v31 = *(_WORD *)(v30 + 32);
  HIBYTE(v32) = 1;
  v33 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 )
  {
    v41 = *(_QWORD *)(v30 + 8) + v48;
    v42 = *(_BYTE *)(v30 + 20);
    if ( (*(_QWORD *)v30 & 4) != 0 )
    {
      *((_QWORD *)&v60 + 1) = *(_QWORD *)(v30 + 8) + v48;
      BYTE4(v61) = v42;
      *(_QWORD *)&v60 = v33 | 4;
      LODWORD(v61) = *(_DWORD *)(v33 + 12);
    }
    else
    {
      *(_QWORD *)&v60 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)&v60 + 1) = v41;
      BYTE4(v61) = v42;
      v44 = *(_QWORD *)(v33 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 <= 1 )
        v44 = **(_QWORD **)(v44 + 16);
      LODWORD(v61) = *(_DWORD *)(v44 + 8) >> 8;
    }
  }
  else
  {
    v34 = *(_DWORD *)(v30 + 16);
    v35 = *(_QWORD *)(v30 + 8) + v48;
    *(_QWORD *)&v60 = 0;
    *((_QWORD *)&v60 + 1) = v35;
    LODWORD(v61) = v34;
    BYTE4(v61) = 0;
  }
  v36 = sub_33F1F00(v28, v56, v57, (__int64)&v54, v19, v49, (__int64)v53, v52, v60, v61, v32, v31, (__int64)&v62, 0);
  *(_QWORD *)a4 = v36;
  *(_DWORD *)(a4 + 8) = v37;
  *((_QWORD *)&v46 + 1) = 1;
  *(_QWORD *)&v46 = v36;
  *((_QWORD *)&v45 + 1) = 1;
  *(_QWORD *)&v45 = a3->m128i_i64[0];
  v38 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v54, 1, 0, 0xFFFFFFFF00000000LL, v45, v46);
  v51 = v49 & 0xFFFFFFFF00000000LL | v39;
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) == 1 || v47 == 16 )
  {
    v40 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = v40.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v40.m128i_i32[2];
  }
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v38, v51);
  if ( v54 )
    sub_B91220((__int64)&v54, v54);
}
