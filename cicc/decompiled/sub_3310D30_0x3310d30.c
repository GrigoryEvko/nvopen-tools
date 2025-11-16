// Function: sub_3310D30
// Address: 0x3310d30
//
__int64 __fastcall sub_3310D30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r15
  __m128i v10; // xmm1
  __int64 v11; // rax
  unsigned __int16 *v12; // rax
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // r9d
  int v20; // edx
  int v21; // r13d
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // r15
  char v26; // al
  __int64 v27; // rdi
  __int64 v28; // rax
  int v29; // edx
  int v30; // r9d
  __int64 v31; // rax
  int v32; // edx
  int v33; // r13d
  int v34; // r9d
  __int64 v35; // rax
  int v36; // edx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r13
  __int128 v41; // rax
  int v42; // r9d
  __int64 v43; // rbx
  int v44; // edx
  __int64 v45; // rax
  int v46; // edx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int128 *v51; // r15
  __int64 v52; // r12
  __int128 v53; // rax
  int v54; // r9d
  __int64 v55; // [rsp+8h] [rbp-A8h]
  int v56; // [rsp+14h] [rbp-9Ch]
  __int64 v57; // [rsp+18h] [rbp-98h]
  int v58; // [rsp+20h] [rbp-90h]
  unsigned __int16 v59; // [rsp+26h] [rbp-8Ah]
  __int64 v60; // [rsp+28h] [rbp-88h]
  __int64 v61; // [rsp+28h] [rbp-88h]
  __int64 v62; // [rsp+28h] [rbp-88h]
  __m128i v63; // [rsp+30h] [rbp-80h]
  __int128 *v64; // [rsp+40h] [rbp-70h]
  __int64 v65; // [rsp+50h] [rbp-60h] BYREF
  int v66; // [rsp+58h] [rbp-58h]
  __int64 v67; // [rsp+60h] [rbp-50h] BYREF
  int v68; // [rsp+68h] [rbp-48h]
  __int64 v69; // [rsp+70h] [rbp-40h]
  int v70; // [rsp+78h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)v8;
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = *(unsigned int *)(v8 + 8);
  v56 = v11;
  v55 = v11;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + 16 * v11);
  v13 = *v12;
  v63 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v57 = *((_QWORD *)v12 + 1);
  v58 = *(_DWORD *)(a2 + 24);
  v14 = *(_QWORD *)(a2 + 48);
  v15 = *(_QWORD *)(a2 + 80);
  v16 = *(unsigned __int16 *)(v14 + 16);
  v17 = *(_QWORD *)(v14 + 24);
  v65 = v15;
  v59 = v16;
  v60 = v17;
  if ( v15 )
    sub_B96E90((__int64)&v65, v15, 1);
  v66 = *(_DWORD *)(a2 + 72);
  if ( !(unsigned __int8)sub_33CF8A0(a2, 1, a3, v16, a5, a6) )
  {
    v18 = *a1;
    v67 = 0;
    v68 = 0;
    v61 = sub_33F17F0(v18, 51, &v67, v59, v60);
    v21 = v20;
    if ( v67 )
      sub_B91220((__int64)&v67, v67);
    v22 = sub_3406EB0(*a1, 56, (unsigned int)&v65, v13, v57, v19, *(_OWORD *)&v63, *(_OWORD *)&v10);
    v70 = v21;
    v67 = v22;
    v68 = v23;
    v69 = v61;
    goto LABEL_7;
  }
  if ( (unsigned __int8)sub_33E2390(*a1, v63.m128i_i64[0], v63.m128i_i64[1], 1)
    && !(unsigned __int8)sub_33E2390(*a1, v10.m128i_i64[0], v10.m128i_i64[1], 1) )
  {
    v24 = sub_3411F20(
            *a1,
            *(_DWORD *)(a2 + 24),
            (unsigned int)&v65,
            *(_QWORD *)(a2 + 48),
            *(_DWORD *)(a2 + 68),
            v30,
            *(_OWORD *)&v10,
            *(_OWORD *)&v63);
    goto LABEL_8;
  }
  v26 = sub_33E0720(v10.m128i_i64[0], v10.m128i_i64[1], 0);
  v27 = *a1;
  if ( v26 )
  {
    v28 = sub_3400BD0(v27, 0, (unsigned int)&v65, v59, v60, 0, 0);
    v70 = v29;
    v67 = v9;
    v68 = v56;
    v69 = v28;
    v24 = sub_32EB790((__int64)a1, a2, &v67, 2, 1);
    goto LABEL_8;
  }
  if ( v58 == 76 )
  {
    if ( !(unsigned int)sub_33DF4A0(v27, v9, v55, v10.m128i_i64[0], v10.m128i_u32[2]) )
    {
LABEL_18:
      v31 = sub_3400BD0(*a1, 0, (unsigned int)&v65, v59, v60, 0, 0);
      v33 = v32;
      v62 = v31;
      v35 = sub_3406EB0(*a1, 56, (unsigned int)&v65, v13, v57, v34, *(_OWORD *)&v63, *(_OWORD *)&v10);
      v70 = v33;
      v67 = v35;
      v68 = v36;
      v69 = v62;
LABEL_7:
      v24 = sub_32EB790((__int64)a1, a2, &v67, 2, 1);
      goto LABEL_8;
    }
    if ( (unsigned __int8)sub_33DFCF0(v63.m128i_i64[0], v63.m128i_i64[1], 0)
      && (unsigned __int8)sub_33E0780(v10.m128i_i64[0], v10.m128i_i64[1], 0, v48, v49, v50) )
    {
      v51 = *(__int128 **)(v9 + 40);
      v52 = *a1;
      *(_QWORD *)&v53 = sub_3400BD0(v52, 0, (unsigned int)&v65, v13, v57, 0, 0);
      v24 = sub_3411F20(v52, 78, (unsigned int)&v65, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v54, v53, *v51);
      goto LABEL_8;
    }
LABEL_27:
    v24 = 0;
    goto LABEL_8;
  }
  if ( !(unsigned int)sub_33DD440(v27, v9, v55, v10.m128i_i64[0], v10.m128i_u32[2], v10.m128i_i64[0]) )
    goto LABEL_18;
  if ( (unsigned __int8)sub_33DFCF0(v63.m128i_i64[0], v63.m128i_i64[1], 0)
    && (unsigned __int8)sub_33E0780(v10.m128i_i64[0], v10.m128i_i64[1], 0, v37, v38, v39) )
  {
    v40 = *a1;
    v64 = *(__int128 **)(v9 + 40);
    *(_QWORD *)&v41 = sub_3400BD0(*a1, 0, (unsigned int)&v65, v13, v57, 0, 0);
    v43 = sub_3411F20(v40, 79, (unsigned int)&v65, *(_QWORD *)(a2 + 48), *(_DWORD *)(a2 + 68), v42, v41, *v64);
    LODWORD(v40) = v44;
    v45 = sub_3407510(
            *a1,
            &v65,
            v43,
            1,
            *(unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16LL),
            *(_QWORD *)(*(_QWORD *)(v43 + 48) + 24LL));
    v67 = v43;
    v68 = v40;
    v69 = v45;
    v70 = v46;
    goto LABEL_7;
  }
  v24 = sub_3271BF0(a1, v63.m128i_i64[0], v63.m128i_i64[1], v10.m128i_i64[0], v10.m128i_u64[1], a2);
  if ( !v24 )
  {
    v47 = sub_3271BF0(a1, v10.m128i_i64[0], v10.m128i_i64[1], v63.m128i_i64[0], v63.m128i_u64[1], a2);
    if ( v47 )
    {
      v24 = v47;
      goto LABEL_8;
    }
    goto LABEL_27;
  }
LABEL_8:
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
  return v24;
}
