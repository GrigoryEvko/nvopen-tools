// Function: sub_3778950
// Address: 0x3778950
//
void __fastcall sub_3778950(__int64 *a1, __int64 a2, __m128i *a3, __int64 a4, __m128i a5)
{
  __int16 *v7; // rax
  __int64 v8; // rsi
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned int *v14; // rax
  __int64 v15; // r13
  __int16 v16; // ax
  __m128i v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  int v20; // r9d
  unsigned __int8 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int32 v24; // edx
  int v25; // r9d
  int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // r13
  __int16 v29; // ax
  __int64 v30; // rdx
  int v31; // r9d
  unsigned __int8 *v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r8
  __int32 v35; // edx
  int v36; // r9d
  int v37; // edx
  __int64 v38; // rsi
  __int64 v39; // r10
  int v40; // eax
  unsigned int *v41; // r15
  __int64 v42; // rax
  __int64 v43; // rax
  __int16 v44; // dx
  __int64 v45; // rax
  __m128i v46; // xmm4
  int v47; // r9d
  unsigned __int8 *v48; // rax
  __int64 v49; // r8
  __int32 v50; // edx
  int v51; // r9d
  int v52; // edx
  unsigned int v53; // r15d
  __m128i v54; // rax
  unsigned int v55; // esi
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r12
  __int64 v60; // rcx
  __int16 v61; // ax
  unsigned __int8 *v62; // rax
  __int64 v63; // rdx
  int v64; // r9d
  unsigned __int8 *v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r8
  __int32 v68; // edx
  int v69; // r9d
  unsigned __int8 *v70; // rax
  int v71; // edx
  __int64 v72; // [rsp+0h] [rbp-1A0h]
  __int64 v73; // [rsp+8h] [rbp-198h]
  int v74; // [rsp+10h] [rbp-190h]
  __int64 v75; // [rsp+10h] [rbp-190h]
  __int16 v76; // [rsp+18h] [rbp-188h]
  __int64 v77; // [rsp+20h] [rbp-180h]
  unsigned __int64 v78; // [rsp+28h] [rbp-178h]
  unsigned __int64 v79; // [rsp+30h] [rbp-170h]
  __int64 v80; // [rsp+30h] [rbp-170h]
  _QWORD *v81; // [rsp+30h] [rbp-170h]
  __int16 v82; // [rsp+30h] [rbp-170h]
  __m128i v84; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v85; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+100h] [rbp-A0h] BYREF
  int v87; // [rsp+108h] [rbp-98h]
  __int64 v88; // [rsp+110h] [rbp-90h] BYREF
  int v89; // [rsp+118h] [rbp-88h]
  __m128i v90; // [rsp+120h] [rbp-80h] BYREF
  __m128i v91; // [rsp+130h] [rbp-70h] BYREF
  __int64 v92; // [rsp+140h] [rbp-60h] BYREF
  __int64 v93; // [rsp+148h] [rbp-58h]
  __m128i v94; // [rsp+150h] [rbp-50h] BYREF
  __m128i v95[4]; // [rsp+160h] [rbp-40h] BYREF

  v7 = *(__int16 **)(a2 + 48);
  v84.m128i_i16[0] = 0;
  v84.m128i_i64[1] = 0;
  v8 = a1[1];
  v85.m128i_i64[1] = 0;
  v85.m128i_i16[0] = 0;
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  LOWORD(v92) = v9;
  v93 = v10;
  sub_33D0340((__int64)&v94, v8, &v92);
  v11 = *(_QWORD *)(a2 + 80);
  v84 = _mm_loadu_si128(&v94);
  v86 = v11;
  v85 = _mm_loadu_si128(v95);
  if ( v11 )
    sub_B96E90((__int64)&v86, v11, 1);
  v12 = a1[1];
  v13 = *a1;
  v87 = *(_DWORD *)(a2 + 72);
  v14 = *(unsigned int **)(a2 + 40);
  v78 = *(_QWORD *)v14;
  v15 = 16LL * v14[2];
  v79 = *(_QWORD *)v14;
  v77 = *((_QWORD *)v14 + 1);
  sub_2FE6CC0(
    (__int64)&v94,
    v13,
    *(_QWORD *)(v12 + 64),
    *(unsigned __int16 *)(v15 + *(_QWORD *)(*(_QWORD *)v14 + 48LL)),
    *(_QWORD *)(v15 + *(_QWORD *)(*(_QWORD *)v14 + 48LL) + 8));
  if ( v94.m128i_i8[0] == 6 )
  {
    sub_375E8D0((__int64)a1, v78, v77, (__int64)a3, a4);
    v21 = sub_33FAF80(a1[1], 234, (__int64)&v86, v84.m128i_u32[0], v84.m128i_i64[1], v20, a5);
    v22 = v85.m128i_u32[0];
    v23 = v85.m128i_i64[1];
    a3->m128i_i64[0] = (__int64)v21;
    a3->m128i_i32[2] = v24;
    *(_QWORD *)a4 = sub_33FAF80(a1[1], 234, (__int64)&v86, v22, v23, v25, a5);
    *(_DWORD *)(a4 + 8) = v26;
    goto LABEL_21;
  }
  if ( v94.m128i_i8[0] > 6u )
  {
    if ( v94.m128i_i8[0] == 10 )
      sub_C64ED0("Scalarization of scalable vectors is not supported.", 1u);
    goto LABEL_6;
  }
  if ( ((v94.m128i_i8[0] - 2) & 0xFD) != 0 )
  {
LABEL_6:
    v16 = v84.m128i_i16[0];
    goto LABEL_7;
  }
  v16 = v84.m128i_i16[0];
  if ( v85.m128i_i16[0] == v84.m128i_i16[0] )
  {
    if ( !v84.m128i_i16[0] && v84.m128i_i64[1] != v85.m128i_i64[1] )
      goto LABEL_35;
    v28 = *(_QWORD *)(v79 + 48) + v15;
    v29 = *(_WORD *)v28;
    v30 = *(_QWORD *)(v28 + 8);
    v94.m128i_i16[0] = v29;
    v94.m128i_i64[1] = v30;
    if ( v29 )
    {
      if ( (unsigned __int16)(v29 - 2) <= 7u
        || (unsigned __int16)(v29 - 17) <= 0x6Cu
        || (unsigned __int16)(v29 - 176) <= 0x1Fu )
      {
        goto LABEL_30;
      }
    }
    else if ( sub_3007070((__int64)&v94) )
    {
LABEL_30:
      sub_375E510((__int64)a1, v78, v77, (__int64)a3, a4);
LABEL_31:
      if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
      {
        a5 = _mm_loadu_si128(a3);
        a3->m128i_i64[0] = *(_QWORD *)a4;
        a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
        *(_QWORD *)a4 = a5.m128i_i64[0];
        *(_DWORD *)(a4 + 8) = a5.m128i_i32[2];
      }
      v32 = sub_33FAF80(a1[1], 234, (__int64)&v86, v84.m128i_u32[0], v84.m128i_i64[1], v31, a5);
      v33 = v85.m128i_u32[0];
      v34 = v85.m128i_i64[1];
      a3->m128i_i64[0] = (__int64)v32;
      a3->m128i_i32[2] = v35;
      *(_QWORD *)a4 = sub_33FAF80(a1[1], 234, (__int64)&v86, v33, v34, v36, a5);
      *(_DWORD *)(a4 + 8) = v37;
LABEL_21:
      v27 = v86;
      if ( !v86 )
        return;
      goto LABEL_22;
    }
    sub_375E6F0((__int64)a1, v78, v77, (__int64)a3, a4);
    goto LABEL_31;
  }
LABEL_7:
  if ( v16 )
  {
    if ( (unsigned __int16)(v16 - 176) > 0x34u )
      goto LABEL_9;
    goto LABEL_36;
  }
LABEL_35:
  if ( sub_3007100((__int64)&v84) )
  {
LABEL_36:
    v38 = *(_QWORD *)(a2 + 80);
    v39 = a1[1];
    v88 = v38;
    if ( v38 )
    {
      v80 = v39;
      sub_B96E90((__int64)&v88, v38, 1);
      v39 = v80;
    }
    v40 = *(_DWORD *)(a2 + 72);
    v41 = *(unsigned int **)(a2 + 40);
    v90.m128i_i64[1] = 0;
    v89 = v40;
    v91.m128i_i64[1] = 0;
    v42 = v41[2];
    v90.m128i_i16[0] = 0;
    v91.m128i_i16[0] = 0;
    v81 = (_QWORD *)v39;
    v43 = *(_QWORD *)(*(_QWORD *)v41 + 48LL) + 16 * v42;
    v44 = *(_WORD *)v43;
    v45 = *(_QWORD *)(v43 + 8);
    LOWORD(v92) = v44;
    v93 = v45;
    sub_33D0340((__int64)&v94, v39, &v92);
    v46 = _mm_loadu_si128(v95);
    v90 = _mm_loadu_si128(&v94);
    v91 = v46;
    sub_3408290((__int64)&v94, v81, (__int128 *)v41, (__int64)&v88, (unsigned int *)&v90, (unsigned int *)&v91, a5);
    if ( v88 )
      sub_B91220((__int64)&v88, v88);
    v48 = sub_33FAF80(a1[1], 234, (__int64)&v86, v84.m128i_u32[0], v84.m128i_i64[1], v47, a5);
    v49 = v85.m128i_i64[1];
    a3->m128i_i64[0] = (__int64)v48;
    a3->m128i_i32[2] = v50;
    *(_QWORD *)a4 = sub_33FAF80(a1[1], 234, (__int64)&v86, v85.m128i_u32[0], v49, v51, a5);
    *(_DWORD *)(a4 + 8) = v52;
    goto LABEL_21;
  }
LABEL_9:
  v17.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v84);
  v94 = v17;
  v18 = sub_CA1930(&v94);
  switch ( v18 )
  {
    case 1u:
      v76 = 2;
      break;
    case 2u:
      v76 = 3;
      break;
    case 4u:
      v76 = 4;
      break;
    case 8u:
      v76 = 5;
      break;
    case 0x10u:
      v76 = 6;
      break;
    case 0x20u:
      v76 = 7;
      break;
    case 0x40u:
      v76 = 8;
      break;
    case 0x80u:
      v76 = 9;
      break;
    default:
      v74 = sub_3007020(*(_QWORD **)(a1[1] + 64), v18);
      v76 = v74;
      v73 = v19;
      goto LABEL_43;
  }
  v73 = 0;
LABEL_43:
  HIWORD(v53) = HIWORD(v74);
  v54.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v85);
  v94 = v54;
  v55 = sub_CA1930(&v94);
  v56 = a1[1];
  switch ( v55 )
  {
    case 1u:
      v82 = 2;
      break;
    case 2u:
      v82 = 3;
      break;
    case 4u:
      v82 = 4;
      break;
    case 8u:
      v82 = 5;
      break;
    case 0x10u:
      v82 = 6;
      break;
    case 0x20u:
      v82 = 7;
      break;
    case 0x40u:
      v82 = 8;
      break;
    case 0x80u:
      v82 = 9;
      break;
    default:
      v72 = sub_3007020(*(_QWORD **)(v56 + 64), v55);
      v82 = v72;
      v56 = a1[1];
      v75 = v57;
      goto LABEL_54;
  }
  v75 = 0;
LABEL_54:
  v58 = v72;
  LOWORD(v58) = v82;
  v59 = v58;
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(v56 + 40)) )
  {
    v60 = v75;
    HIWORD(v53) = WORD1(v59);
    v75 = v73;
    v61 = v76;
    v73 = v60;
    v76 = v82;
    v82 = v61;
  }
  v62 = sub_375A6A0((__int64)a1, v78, v77, a5);
  LOWORD(v59) = v82;
  LOWORD(v53) = v76;
  sub_375B6E0(a1, (__int64)v62, v63, v53, v73, (__int64)a3, a5, v59, v75, a4);
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
  {
    a5 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = a5.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = a5.m128i_i32[2];
  }
  v65 = sub_33FAF80(a1[1], 234, (__int64)&v86, v84.m128i_u32[0], v84.m128i_i64[1], v64, a5);
  v66 = v85.m128i_u32[0];
  v67 = v85.m128i_i64[1];
  a3->m128i_i64[0] = (__int64)v65;
  a3->m128i_i32[2] = v68;
  v70 = sub_33FAF80(a1[1], 234, (__int64)&v86, v66, v67, v69, a5);
  v27 = v86;
  *(_QWORD *)a4 = v70;
  *(_DWORD *)(a4 + 8) = v71;
  if ( v27 )
LABEL_22:
    sub_B91220((__int64)&v86, v27);
}
