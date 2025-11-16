// Function: sub_383DA20
// Address: 0x383da20
//
unsigned __int8 *__fastcall sub_383DA20(__int64 *a1, unsigned __int64 a2, int a3, __m128i a4)
{
  unsigned __int64 v4; // r13
  const __m128i *v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rdx
  __int64 (*v9)(void); // rax
  unsigned int v10; // edx
  __int64 v11; // r9
  __int64 v12; // r12
  unsigned __int16 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // esi
  const __m128i *v17; // r9
  unsigned __int64 v18; // r10
  __int64 v19; // r11
  __int128 *v20; // r8
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned __int16 v23; // cx
  __int64 *v24; // r12
  unsigned int i; // ebx
  __int64 v26; // rdx
  unsigned __int8 *result; // rax
  unsigned __int16 *v28; // rax
  __int64 v29; // rdx
  unsigned int v30; // r13d
  __int64 v31; // r15
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // r10
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 (__fastcall *v37)(__int64, __int64, unsigned int, __int64); // r13
  __int64 v38; // rax
  unsigned __int16 v39; // si
  __int64 v40; // r8
  __int64 v41; // rax
  unsigned int v42; // r15d
  unsigned __int16 v43; // r13
  unsigned __int16 v44; // r11
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdx
  __int128 *v48; // r12
  unsigned __int64 v49; // r10
  const __m128i *v50; // r9
  __int64 v51; // r11
  __int64 v52; // rsi
  __int64 v53; // r8
  unsigned __int16 v54; // cx
  __int64 *v55; // r12
  __int64 v56; // rsi
  __int64 v57; // rbx
  int v58; // eax
  int v59; // eax
  __int64 v60; // rdx
  unsigned __int64 v61; // [rsp+0h] [rbp-C0h]
  __int64 v62; // [rsp+8h] [rbp-B8h]
  unsigned __int16 v63; // [rsp+10h] [rbp-B0h]
  const __m128i *v64; // [rsp+18h] [rbp-A8h]
  unsigned __int16 v65; // [rsp+18h] [rbp-A8h]
  __int128 *v66; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v67; // [rsp+20h] [rbp-A0h]
  __int64 v68; // [rsp+28h] [rbp-98h]
  __int128 *v69; // [rsp+30h] [rbp-90h]
  __int64 v70; // [rsp+30h] [rbp-90h]
  __int64 v71; // [rsp+38h] [rbp-88h]
  const __m128i *v72; // [rsp+38h] [rbp-88h]
  __int128 v73; // [rsp+40h] [rbp-80h]
  __int64 v74; // [rsp+40h] [rbp-80h]
  _QWORD *v75; // [rsp+40h] [rbp-80h]
  _QWORD *v76; // [rsp+50h] [rbp-70h]
  __int64 v77; // [rsp+50h] [rbp-70h]
  __int64 v78; // [rsp+58h] [rbp-68h]
  unsigned int v79; // [rsp+60h] [rbp-60h]
  __int64 (__fastcall *v80)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+60h] [rbp-60h]
  unsigned __int16 v81; // [rsp+60h] [rbp-60h]
  __int128 *v82; // [rsp+60h] [rbp-60h]
  unsigned __int8 *v83; // [rsp+60h] [rbp-60h]
  __int64 v84; // [rsp+70h] [rbp-50h] BYREF
  int v85; // [rsp+78h] [rbp-48h]
  __int64 v86; // [rsp+80h] [rbp-40h]

  v6 = *(const __m128i **)(a2 + 40);
  if ( a3 != 1 )
  {
    v7 = _mm_loadu_si128(v6 + 5);
    *(_QWORD *)&v73 = sub_37AE0F0((__int64)a1, v6[7].m128i_u64[1], v6[8].m128i_i64[0]);
    *((_QWORD *)&v73 + 1) = v8;
    v9 = *(__int64 (**)(void))(*(_QWORD *)*a1 + 1216LL);
    if ( v9 != sub_2FE3360 )
    {
      v58 = v9();
      if ( v58 == 214 )
      {
        v12 = (__int64)sub_37AF270((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1], v7);
        goto LABEL_4;
      }
      if ( v58 != 215 )
      {
        if ( v58 != 213 )
          BUG();
        v12 = (__int64)sub_383B380((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1]);
        goto LABEL_4;
      }
    }
    v12 = sub_37AE0F0((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1]);
LABEL_4:
    v71 = v10;
    v13 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v10);
    v14 = sub_33E5B50(
            (_QWORD *)a1[1],
            *v13,
            *((_QWORD *)v13 + 1),
            *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL),
            v11,
            1,
            0);
    v16 = *(_DWORD *)(a2 + 24);
    v17 = *(const __m128i **)(a2 + 112);
    v18 = v14;
    v19 = v15;
    v20 = *(__int128 **)(a2 + 40);
    v76 = (_QWORD *)a1[1];
    if ( v16 == 339 )
      v69 = v20 + 5;
    else
      v69 = (__int128 *)((char *)v20 + 40);
    v21 = *(_QWORD *)(a2 + 80);
    v22 = *(_QWORD *)(a2 + 104);
    v23 = *(_WORD *)(a2 + 96);
    v84 = v21;
    if ( v21 )
    {
      v63 = v23;
      v61 = v18;
      v62 = v15;
      v64 = v17;
      v66 = v20;
      sub_B96E90((__int64)&v84, v21, 1);
      v16 = *(_DWORD *)(a2 + 24);
      v23 = v63;
      v18 = v61;
      v19 = v62;
      v17 = v64;
      v20 = v66;
    }
    v85 = *(_DWORD *)(a2 + 72);
    v24 = sub_33E6F00(
            v76,
            v16,
            (__int64)&v84,
            v23,
            v22,
            v17,
            v18,
            v19,
            *v20,
            *v69,
            __PAIR128__(v71 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL, v12),
            v73);
    if ( v84 )
      sub_B91220((__int64)&v84, v84);
    v79 = *(_DWORD *)(a2 + 68);
    if ( v79 > 1 )
    {
      for ( i = 1; i != v79; ++i )
      {
        v26 = i;
        v4 = v26 | v4 & 0xFFFFFFFF00000000LL;
        sub_3760E70((__int64)a1, a2, v26, (unsigned __int64)v24, v4);
      }
    }
    return (unsigned __int8 *)v24;
  }
  v35 = *a1;
  v28 = (unsigned __int16 *)(*(_QWORD *)(v6[5].m128i_i64[0] + 48) + 16LL * v6[5].m128i_u32[2]);
  v29 = a1[1];
  v30 = *v28;
  v31 = *(_QWORD *)(v29 + 64);
  v78 = *((_QWORD *)v28 + 1);
  v80 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)*a1 + 528LL);
  v32 = sub_2E79000(*(__int64 **)(v29 + 40));
  v33 = v80(v35, v32, v31, v30, v78);
  v34 = *a1;
  WORD1(v35) = HIWORD(v33);
  v74 = v36;
  v81 = v33;
  v37 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v38 = *(_QWORD *)(a2 + 48);
  v39 = *(_WORD *)(v38 + 16);
  v40 = *(_QWORD *)(v38 + 24);
  v41 = a1[1];
  if ( v37 == sub_2D56A50 )
  {
    HIWORD(v42) = 0;
    sub_2FE6CC0((__int64)&v84, v34, *(_QWORD *)(v41 + 64), v39, v40);
    v43 = v85;
    v44 = v81;
    v45 = v74;
    v77 = v86;
  }
  else
  {
    v59 = v37(v34, *(_QWORD *)(v41 + 64), v39, v40);
    v45 = v74;
    v44 = v81;
    HIWORD(v42) = HIWORD(v59);
    v43 = v59;
    v77 = v60;
  }
  if ( !v44 || !*(_QWORD *)(*a1 + 8LL * v44 + 112) )
  {
    v45 = v77;
    v44 = v43;
    WORD1(v35) = HIWORD(v42);
  }
  LOWORD(v35) = v44;
  v46 = sub_33E5B50(
          (_QWORD *)a1[1],
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          (unsigned int)v35,
          v45,
          v45,
          1,
          0);
  v48 = *(__int128 **)(a2 + 40);
  v49 = v46;
  v50 = *(const __m128i **)(a2 + 112);
  v51 = v47;
  v75 = (_QWORD *)a1[1];
  v82 = v48 + 5;
  if ( *(_DWORD *)(a2 + 24) != 339 )
    v82 = (__int128 *)((char *)v48 + 40);
  v52 = *(_QWORD *)(a2 + 80);
  v53 = *(_QWORD *)(a2 + 104);
  v54 = *(_WORD *)(a2 + 96);
  v84 = v52;
  if ( v52 )
  {
    v65 = v54;
    v67 = v46;
    v68 = v47;
    v70 = v53;
    v72 = v50;
    sub_B96E90((__int64)&v84, v52, 1);
    v54 = v65;
    v49 = v67;
    v51 = v68;
    v53 = v70;
    v50 = v72;
  }
  v85 = *(_DWORD *)(a2 + 72);
  v55 = sub_33E6F00(
          v75,
          341,
          (__int64)&v84,
          v54,
          v53,
          v50,
          v49,
          v51,
          *v48,
          *v82,
          v48[5],
          *(__int128 *)((char *)v48 + 120));
  if ( v84 )
    sub_B91220((__int64)&v84, v84);
  sub_3760E70((__int64)a1, a2, 0, (unsigned __int64)v55, 0);
  sub_3760E70((__int64)a1, a2, 2, (unsigned __int64)v55, 2);
  v56 = *(_QWORD *)(a2 + 80);
  v57 = a1[1];
  v84 = v56;
  if ( v56 )
    sub_B96E90((__int64)&v84, v56, 1);
  LOWORD(v42) = v43;
  v85 = *(_DWORD *)(a2 + 72);
  result = sub_33FB160(v57, (__int64)v55, 1u, (__int64)&v84, v42, v77, a4);
  if ( v84 )
  {
    v83 = result;
    sub_B91220((__int64)&v84, v84);
    return v83;
  }
  return result;
}
