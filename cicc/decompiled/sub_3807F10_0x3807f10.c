// Function: sub_3807F10
// Address: 0x3807f10
//
_QWORD *__fastcall sub_3807F10(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int16 *v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // r9
  __int64 v13; // r8
  __m128i v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r8
  __int64 v17; // r14
  int v18; // eax
  unsigned __int64 v19; // rax
  _WORD *v20; // rsi
  __m128i v21; // xmm0
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r10
  __int64 *v26; // rdi
  __m128i *v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // r12
  __int64 v31; // rdi
  _QWORD *v32; // rdi
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 (__fastcall *v36)(__int64, __int64, unsigned int); // rdx
  __int64 v37; // [rsp+8h] [rbp-158h]
  __int64 v38; // [rsp+8h] [rbp-158h]
  __int64 (__fastcall *v39)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-150h]
  int v40; // [rsp+18h] [rbp-148h]
  unsigned __int16 v41; // [rsp+1Eh] [rbp-142h]
  __m128i v42; // [rsp+20h] [rbp-140h] BYREF
  unsigned int v43; // [rsp+30h] [rbp-130h] BYREF
  __int64 v44; // [rsp+38h] [rbp-128h]
  __int64 v45; // [rsp+40h] [rbp-120h] BYREF
  int v46; // [rsp+48h] [rbp-118h]
  __int64 v47; // [rsp+50h] [rbp-110h]
  __int64 v48; // [rsp+58h] [rbp-108h]
  __int128 v49; // [rsp+60h] [rbp-100h] BYREF
  __int64 v50; // [rsp+70h] [rbp-F0h]
  _QWORD v51[2]; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v52; // [rsp+90h] [rbp-D0h]
  unsigned __int16 v53; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-B8h]
  __int16 v55; // [rsp+B0h] [rbp-B0h]
  __int64 v56; // [rsp+B8h] [rbp-A8h]
  _QWORD v57[4]; // [rsp+C0h] [rbp-A0h] BYREF
  _QWORD v58[4]; // [rsp+E0h] [rbp-80h] BYREF
  const char *v59; // [rsp+100h] [rbp-60h] BYREF
  __int64 v60; // [rsp+108h] [rbp-58h]
  __int64 (__fastcall *v61)(__int64, __int64, unsigned int); // [rsp+110h] [rbp-50h]
  __int64 v62; // [rsp+118h] [rbp-48h]
  __int64 v63; // [rsp+120h] [rbp-40h]

  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *((_QWORD *)v4 + 1);
  v6 = *((_QWORD *)v4 + 3);
  v7 = *v4;
  v41 = *v4;
  LOWORD(v43) = v4[8];
  v44 = v6;
  v40 = sub_2FE5ED0(v41, v5);
  v42.m128i_i64[0] = *(unsigned int *)(**(_QWORD **)(a1[1] + 24) + 172LL);
  if ( (_WORD)v43 )
  {
    if ( (_WORD)v43 == 1 || (unsigned __int16)(v43 - 504) <= 7u )
      BUG();
    v9 = 16LL * ((unsigned __int16)v43 - 1);
    v8 = *(_QWORD *)&byte_444C4A0[v9];
    LOBYTE(v9) = byte_444C4A0[v9 + 8];
  }
  else
  {
    v8 = sub_3007260((__int64)&v43);
    v47 = v8;
    v48 = v9;
  }
  v59 = (const char *)v8;
  LOBYTE(v60) = v9;
  if ( v42.m128i_i64[0] == sub_CA1930(&v59) )
  {
    v10 = a1[1];
    v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v11 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v59, *a1, *(_QWORD *)(v10 + 64), v41, v5);
      v13 = (unsigned __int16)v60;
      v39 = v61;
    }
    else
    {
      v35 = v11(*a1, *(_QWORD *)(v10 + 64), v7, v5);
      v39 = v36;
      v13 = v35;
    }
    v37 = v13;
    v14.m128i_i64[0] = (__int64)sub_33EDFE0(a1[1], v43, v44, 1, v13, v12);
    v15 = *(_QWORD *)(a2 + 80);
    v16 = v37;
    v42 = v14;
    v17 = v14.m128i_i64[0];
    v45 = v15;
    if ( v15 )
    {
      sub_B96E90((__int64)&v45, v15, 1);
      v16 = v37;
    }
    v18 = *(_DWORD *)(a2 + 72);
    LOBYTE(v63) = 4;
    v38 = v16;
    v46 = v18;
    v19 = sub_3805E70((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
    v20 = (_WORD *)*a1;
    v54 = v5;
    v51[0] = v19;
    v21 = _mm_load_si128(&v42);
    v51[1] = v22;
    v23 = *(_QWORD *)(v17 + 48) + 16LL * v42.m128i_u32[2];
    v53 = v41;
    v52 = v21;
    LOWORD(v22) = *(_WORD *)v23;
    v24 = *(_QWORD *)(v23 + 8);
    v62 = v5;
    v25 = a1[1];
    v55 = v22;
    v56 = v24;
    LOWORD(v61) = v41;
    LOBYTE(v63) = v63 | 0x10;
    v59 = (const char *)&v53;
    v60 = 2;
    sub_3494590(
      (__int64)v57,
      v20,
      v25,
      v40,
      v38,
      v39,
      (__int64)v51,
      2u,
      (__int64)&v53,
      2,
      (unsigned int)v61,
      v5,
      v63,
      (__int64)&v45,
      0,
      0);
    sub_2EAC300((__int64)&v49, *(_QWORD *)(a1[1] + 40), *(_DWORD *)(v17 + 96), 0);
    v26 = (__int64 *)a1[1];
    memset(v58, 0, sizeof(v58));
    v27 = sub_33F1F00(
            v26,
            v43,
            v44,
            (__int64)&v45,
            v57[2],
            v57[3],
            v42.m128i_i64[0],
            v42.m128i_i64[1],
            v49,
            v50,
            0,
            0,
            (__int64)v58,
            0);
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v27, v28);
    v29 = (_QWORD *)v57[0];
    if ( v45 )
      sub_B91220((__int64)&v45, v45);
  }
  else
  {
    v31 = *(_QWORD *)(a1[1] + 64);
    v59 = "ffrexp exponent does not match sizeof(int)";
    LOWORD(v63) = 259;
    sub_B6ECE0(v31, (__int64)&v59);
    v32 = (_QWORD *)a1[1];
    v33 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v34 = **(unsigned __int16 **)(a2 + 48);
    v59 = 0;
    LODWORD(v60) = 0;
    v29 = sub_33F17F0(v32, 51, (__int64)&v59, v34, v33);
    if ( v59 )
      sub_B91220((__int64)&v59, (__int64)v59);
  }
  return v29;
}
