// Function: sub_2132880
// Address: 0x2132880
//
unsigned __int64 __fastcall sub_2132880(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i *a4,
        __m128i a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v7; // r9
  char *v10; // rax
  __int64 v11; // rsi
  unsigned __int8 v12; // bl
  __int64 v13; // rbx
  __int64 v14; // rdx
  int v15; // eax
  unsigned __int64 result; // rax
  unsigned __int8 v17; // r14
  __int64 v18; // r9
  int v19; // eax
  __m128i *v20; // rsi
  __int8 v21; // dl
  unsigned __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r14
  unsigned __int8 *v25; // rax
  unsigned int v26; // ebx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rax
  const void **v31; // rdx
  const void **v32; // r9
  __int64 v33; // rdx
  __int64 v34; // r10
  __int64 *v35; // r15
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  const void ***v38; // rax
  int v39; // edx
  __int64 v40; // r9
  __int64 *v41; // rax
  __int64 v42; // rsi
  int v43; // edx
  int v44; // eax
  __int64 v45; // rax
  int v46; // ecx
  unsigned int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rdx
  __m128i v50; // xmm2
  __int128 v51; // [rsp-10h] [rbp-100h]
  unsigned __int64 v52; // [rsp-10h] [rbp-100h]
  __int64 v53; // [rsp+8h] [rbp-E8h]
  unsigned int v54; // [rsp+14h] [rbp-DCh]
  __int64 v55; // [rsp+20h] [rbp-D0h]
  __int64 v56; // [rsp+20h] [rbp-D0h]
  __int64 v57; // [rsp+30h] [rbp-C0h]
  __int64 v58; // [rsp+30h] [rbp-C0h]
  unsigned int v59; // [rsp+30h] [rbp-C0h]
  unsigned __int8 v60; // [rsp+38h] [rbp-B8h]
  __int64 v61; // [rsp+38h] [rbp-B8h]
  __int64 v62; // [rsp+50h] [rbp-A0h] BYREF
  int v63; // [rsp+58h] [rbp-98h]
  __m128i v64; // [rsp+60h] [rbp-90h] BYREF
  __m128i v65[2]; // [rsp+70h] [rbp-80h] BYREF
  _OWORD v66[2]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v67; // [rsp+B0h] [rbp-40h]
  unsigned __int64 v68; // [rsp+B8h] [rbp-38h]

  v7 = a2;
  v10 = *(char **)(a2 + 40);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *v10;
  v62 = v11;
  v60 = v12;
  v13 = *((_QWORD *)v10 + 1);
  if ( v11 )
  {
    v57 = v7;
    sub_1623A60((__int64)&v62, v11, 2);
    v7 = v57;
  }
  v63 = *(_DWORD *)(v7 + 64);
  v14 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL);
  v15 = *(unsigned __int16 *)(v14 + 24);
  if ( v15 == 32 || v15 == 10 )
  {
    result = sub_2129860(
               a1,
               v7,
               (const void ***)(*(_QWORD *)(v14 + 88) + 24LL),
               a3,
               (__int64)a4,
               a5,
               *(double *)a6.m128i_i64,
               a7);
    goto LABEL_18;
  }
  v58 = v7;
  result = sub_212AC20(a1, v7, a3, a4, a5, *(double *)a6.m128i_i64, a7);
  v17 = result;
  if ( (_BYTE)result )
    goto LABEL_18;
  v18 = v58;
  v59 = 139;
  v19 = *(unsigned __int16 *)(v18 + 24);
  if ( v19 != 122 )
    v59 = (v19 == 124) + 140;
  v55 = v18;
  sub_1F40D10((__int64)v66, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v60, v13);
  v20 = *(__m128i **)a1;
  if ( !BYTE8(v66[0]) )
    goto LABEL_21;
  v21 = v20[151].m128i_i8[259 * BYTE8(v66[0]) + 6 + v59];
  if ( !v21 )
  {
    if ( v20[7].m128i_i64[BYTE8(v66[0]) + 1] )
      goto LABEL_11;
    goto LABEL_21;
  }
  if ( v21 != 4 )
  {
LABEL_21:
    v44 = *(unsigned __int16 *)(v55 + 24);
    if ( v44 == 122 )
    {
      switch ( v60 )
      {
        case 4u:
          v45 = 0;
          v46 = 0;
          break;
        case 5u:
          v45 = 1;
          v46 = 1;
          break;
        case 6u:
          v45 = 2;
          v46 = 2;
          break;
        case 7u:
          v45 = 3;
          v46 = 3;
          break;
        default:
          goto LABEL_36;
      }
    }
    else if ( v44 == 124 )
    {
      switch ( v60 )
      {
        case 4u:
          v45 = 4;
          v46 = 4;
          break;
        case 5u:
          v45 = 5;
          v46 = 5;
          break;
        case 6u:
          v45 = 6;
          v46 = 6;
          break;
        case 7u:
          v45 = 7;
          v46 = 7;
          break;
        default:
LABEL_36:
          result = sub_212B970(
                     (__int64 *)a1,
                     v55,
                     a3,
                     (__int64)a4,
                     *(double *)a5.m128i_i64,
                     *(double *)a6.m128i_i64,
                     a7);
          v42 = v62;
          if ( !v62 )
            return result;
          return sub_161E7C0((__int64)&v62, v42);
      }
    }
    else
    {
      switch ( v60 )
      {
        case 4u:
          v45 = 8;
          v17 = 1;
          v46 = 8;
          break;
        case 5u:
          v45 = 9;
          v17 = 1;
          v46 = 9;
          break;
        case 6u:
          v45 = 10;
          v17 = 1;
          v46 = 10;
          break;
        case 7u:
          v45 = 11;
          v17 = 1;
          v46 = 11;
          break;
        default:
          goto LABEL_36;
      }
    }
    if ( !v20[4631].m128i_i64[v45] )
      goto LABEL_36;
    v48 = *(_QWORD *)(v55 + 32);
    v49 = *(_QWORD *)(a1 + 8);
    v50 = _mm_loadu_si128((const __m128i *)v48);
    v65[0] = v50;
    v65[1] = _mm_loadu_si128((const __m128i *)(v48 + 40));
    sub_20BE530((__int64)v66, v20, v49, v46, v60, v13, a5, a6, v50, (__int64)v65, 2u, v17, (__int64)&v62, 0, 1);
    result = sub_200E870(
               a1,
               *(__int64 *)&v66[0],
               *((unsigned __int64 *)&v66[0] + 1),
               a3,
               a4,
               a5,
               *(double *)a6.m128i_i64,
               v50);
LABEL_18:
    v42 = v62;
    if ( !v62 )
      return result;
    return sub_161E7C0((__int64)&v62, v42);
  }
LABEL_11:
  v22 = *(unsigned __int64 **)(v55 + 32);
  v64.m128i_i32[2] = 0;
  v65[0].m128i_i64[0] = 0;
  v65[0].m128i_i32[2] = 0;
  v23 = v22[1];
  v64.m128i_i64[0] = 0;
  sub_20174B0(a1, *v22, v23, &v64, v65);
  v24 = *(_QWORD *)a1;
  v25 = (unsigned __int8 *)(*(_QWORD *)(v64.m128i_i64[0] + 40) + 16LL * v64.m128i_u32[2]);
  v61 = *((_QWORD *)v25 + 1);
  v26 = *v25;
  v27 = *(_QWORD *)(v55 + 32);
  v53 = *(_QWORD *)(v27 + 40);
  v54 = *(_DWORD *)(v27 + 48);
  v56 = *(_QWORD *)(v27 + 48);
  v28 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
  v29 = sub_1F40B60(v24, v26, v61, v28, 1);
  v30 = v54;
  v32 = v31;
  v33 = *(_QWORD *)(v53 + 40) + 16LL * v54;
  if ( *(_BYTE *)v33 != (_BYTE)v29 || (v34 = v53, *(const void ***)(v33 + 8) != v32) && !*(_BYTE *)v33 )
  {
    v34 = sub_1D323C0(
            *(__int64 **)(a1 + 8),
            v53,
            v56,
            (__int64)&v62,
            v29,
            v32,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64);
    v30 = v47;
  }
  v35 = *(__int64 **)(a1 + 8);
  v36 = _mm_loadu_si128(&v64);
  v37 = _mm_loadu_si128(v65);
  v67 = v34;
  v66[0] = v36;
  v68 = v30 | v56 & 0xFFFFFFFF00000000LL;
  v66[1] = v37;
  v38 = (const void ***)sub_1D252B0((__int64)v35, v26, v61, v26, v61);
  *((_QWORD *)&v51 + 1) = 3;
  *(_QWORD *)&v51 = v66;
  v41 = sub_1D36D80(v35, v59, (__int64)&v62, v38, v39, *(double *)v36.m128i_i64, *(double *)v37.m128i_i64, a7, v40, v51);
  v42 = v62;
  *(_QWORD *)a3 = v41;
  *(_DWORD *)(a3 + 8) = v43;
  a4->m128i_i64[0] = (__int64)v41;
  result = v52;
  a4->m128i_i32[2] = 1;
  if ( v42 )
    return sub_161E7C0((__int64)&v62, v42);
  return result;
}
