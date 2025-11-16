// Function: sub_2BED190
// Address: 0x2bed190
//
__int64 __fastcall sub_2BED190(__int64 a1, __m128i *a2, __int64 a3, volatile signed __int32 **a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // r12
  volatile signed __int32 **v10; // rdi
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v21; // r15
  _BYTE *v22; // rsi
  __m128i v23; // xmm3
  __m128i v24; // xmm2
  __m128i v25; // xmm0
  __m128i *v26; // rsi
  __m128i v27; // xmm0
  bool v28; // zf
  __m128i v29; // xmm1
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __m128i *v33; // r14
  __int64 v34; // r14
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // r15
  _QWORD *v39; // r14
  __int64 v40; // rax
  __m128i v41; // xmm0
  __m128i v42; // xmm4
  __m128i v43; // xmm5
  __m128i *v44; // rsi
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __m128i *v50; // r13
  __int64 v51; // r13
  unsigned __int64 *v52; // r14
  __m128i *v53; // rsi
  __m128i v54; // xmm0
  __m128i v55; // xmm1
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 result; // rax
  __int64 v62; // rcx
  unsigned int *v63; // rdi
  unsigned int *i; // rdx
  __int64 j; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 *v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rcx
  volatile signed __int32 **v71; // [rsp+8h] [rbp-138h]
  __int64 v72; // [rsp+8h] [rbp-138h]
  unsigned __int64 v73; // [rsp+8h] [rbp-138h]
  __int64 v74; // [rsp+18h] [rbp-128h] BYREF
  __m128i v75; // [rsp+20h] [rbp-120h] BYREF
  __m128i v76; // [rsp+30h] [rbp-110h] BYREF
  __m128i v77; // [rsp+40h] [rbp-100h] BYREF
  __m128i v78; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v79; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v80; // [rsp+70h] [rbp-D0h] BYREF
  __m128i v81; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v82; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v83; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i v84; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v85; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v86; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v87; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v88; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v89; // [rsp+100h] [rbp-40h] BYREF

  if ( (a5 & 0x3F0) == 0 )
    a5 |= 0x10u;
  *(_DWORD *)a1 = a5;
  sub_2208E20((volatile signed __int32 **)&v87, a4);
  sub_2BDFD50(a1 + 8, (__int64)a2, a3, *(_DWORD *)a1, &v87);
  sub_2209150((volatile signed __int32 **)&v87);
  *(_QWORD *)(a1 + 256) = 0;
  v8 = sub_22077B0(0x68u);
  v9 = v8;
  if ( v8 )
  {
    *(_BYTE *)(v8 + 64) = 0;
    *(_QWORD *)(v8 + 8) = 0x100000001LL;
    v10 = (volatile signed __int32 **)(v8 + 96);
    v71 = (volatile signed __int32 **)(v8 + 96);
    *(_QWORD *)v8 = &unk_4A238A8;
    v11 = *(_DWORD *)a1;
    *(_QWORD *)(v9 + 16) = 0;
    *(_DWORD *)(v9 + 40) = v11;
    *(_QWORD *)(v9 + 24) = 0;
    *(_QWORD *)(v9 + 32) = 0;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = 0;
    *(_QWORD *)(v9 + 72) = 0;
    *(_QWORD *)(v9 + 80) = 0;
    *(_QWORD *)(v9 + 88) = 0;
    sub_220A990(v10);
    sub_2208E20((volatile signed __int32 **)&v84, a4);
    sub_2208E20((volatile signed __int32 **)&v87, v71);
    sub_22090A0(v71, (volatile signed __int32 **)&v84);
    sub_22090A0((volatile signed __int32 **)&v84, (volatile signed __int32 **)&v87);
    sub_2209150((volatile signed __int32 **)&v87);
    a2 = &v84;
    sub_2208E20((volatile signed __int32 **)&v87, (volatile signed __int32 **)&v84);
    sub_2209150((volatile signed __int32 **)&v87);
    sub_2209150((volatile signed __int32 **)&v84);
  }
  *(_QWORD *)(a1 + 264) = v9;
  *(_QWORD *)(a1 + 256) = v9 + 16;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 312) = 8;
  v12 = sub_22077B0(0x40u);
  v13 = *(_QWORD *)(a1 + 312);
  *(_QWORD *)(a1 + 304) = v12;
  v14 = (__int64 *)(v12 + ((4 * v13 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v15 = sub_22077B0(0x1F8u);
  *(_QWORD *)(a1 + 344) = v14;
  v16 = v15 + 504;
  *v14 = v15;
  *(_QWORD *)(a1 + 328) = v15;
  *(_QWORD *)(a1 + 360) = v15;
  *(_QWORD *)(a1 + 320) = v15;
  *(_QWORD *)(a1 + 352) = v15;
  v17 = *(_QWORD *)(a1 + 256);
  *(_QWORD *)(a1 + 336) = v16;
  *(_QWORD *)(a1 + 376) = v14;
  *(_QWORD *)(a1 + 368) = v16;
  *(_QWORD *)(a1 + 384) = v17 + 80;
  v18 = sub_222F790(a4, (__int64)a2);
  v19 = *(_QWORD **)(a1 + 256);
  *(_QWORD *)(a1 + 392) = v18;
  v21 = v19[4];
  v22 = (_BYTE *)v19[1];
  v74 = v19[5];
  v20 = v74;
  v19[5] = v74 + 1;
  if ( v22 == (_BYTE *)v19[2] )
  {
    sub_9CA200((__int64)v19, v22, &v74);
    v20 = v74;
  }
  else
  {
    if ( v22 )
    {
      *(_QWORD *)v22 = v20;
      v22 = (_BYTE *)v19[1];
    }
    v19[1] = v22 + 8;
  }
  v76.m128i_i64[0] = v20;
  v23 = _mm_loadu_si128(&v77);
  v75.m128i_i32[0] = 8;
  v24 = _mm_loadu_si128(&v76);
  v75.m128i_i64[1] = -1;
  v25 = _mm_loadu_si128(&v75);
  v80 = v23;
  v26 = (__m128i *)v19[8];
  v78 = v25;
  v79 = v24;
  if ( v26 == (__m128i *)v19[9] )
  {
    sub_2BE00E0(v19 + 7, v26, &v78);
    v33 = (__m128i *)v19[8];
  }
  else
  {
    if ( v26 )
    {
      *v26 = v25;
      v27 = _mm_loadu_si128(&v79);
      v28 = v78.m128i_i32[0] == 11;
      v26[1] = v27;
      v26[2] = _mm_loadu_si128(&v80);
      if ( v28 )
      {
        v29 = _mm_loadu_si128(&v79);
        v79 = v27;
        v30 = v26[2].m128i_i64[1];
        v26[2].m128i_i64[0] = 0;
        v26[1] = v29;
        v31 = v80.m128i_i64[0];
        v80.m128i_i64[0] = 0;
        v26[2].m128i_i64[0] = v31;
        v32 = v80.m128i_i64[1];
        v80.m128i_i64[1] = v30;
        v26[2].m128i_i64[1] = v32;
      }
      v26 = (__m128i *)v19[8];
    }
    v33 = v26 + 3;
    v19[8] = v26 + 3;
  }
  v34 = (__int64)v33->m128i_i64 - v19[7];
  if ( (unsigned __int64)v34 > 0x493E00 )
    goto LABEL_70;
  v35 = 0xAAAAAAAAAAAAAAABLL * (v34 >> 4) - 1;
  if ( v78.m128i_i32[0] == 11 && v80.m128i_i64[0] )
  {
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v80.m128i_i64[0])(&v79, &v79, 3);
    v35 = 0xAAAAAAAAAAAAAAABLL * (v34 >> 4) - 1;
  }
  if ( v75.m128i_i32[0] == 11 && v77.m128i_i64[0] )
  {
    v73 = v35;
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v77.m128i_i64[0])(&v76, &v76, 3);
    v35 = v73;
  }
  *(_QWORD *)(v19[7] + 48 * v21 + 8) = v35;
  sub_2BECC80(a1);
  if ( *(_DWORD *)(a1 + 152) != 27 || !(unsigned __int8)sub_2BE0030(a1) )
    sub_4392CE(5u);
  v36 = *(_QWORD *)(a1 + 352);
  if ( v36 == *(_QWORD *)(a1 + 360) )
  {
    v67 = *(_QWORD *)(*(_QWORD *)(a1 + 376) - 8LL);
    v38 = *(_QWORD *)(v67 + 496);
    v72 = *(_QWORD *)(v67 + 488);
    j_j___libc_free_0(v36);
    v37 = v72;
    v68 = (__int64 *)(*(_QWORD *)(a1 + 376) - 8LL);
    *(_QWORD *)(a1 + 376) = v68;
    v69 = *v68;
    v70 = *v68 + 504;
    *(_QWORD *)(a1 + 360) = v69;
    *(_QWORD *)(a1 + 368) = v70;
    *(_QWORD *)(a1 + 352) = v69 + 480;
  }
  else
  {
    v37 = *(_QWORD *)(v36 - 16);
    v38 = *(_QWORD *)(v36 - 8);
    *(_QWORD *)(a1 + 352) = v36 - 24;
  }
  *(_QWORD *)(v19[7] + v34 - 40) = v37;
  v39 = *(_QWORD **)(a1 + 256);
  v84.m128i_i32[0] = 9;
  v84.m128i_i64[1] = -1;
  v40 = v39[1];
  v85.m128i_i64[0] = *(_QWORD *)(v40 - 8);
  v39[1] = v40 - 8;
  v41 = _mm_loadu_si128(&v84);
  v42 = _mm_loadu_si128(&v85);
  v43 = _mm_loadu_si128(&v86);
  v87 = v41;
  v88 = v42;
  v89 = v43;
  v44 = (__m128i *)v39[8];
  if ( v44 == (__m128i *)v39[9] )
  {
    sub_2BE00E0(v39 + 7, v44, &v87);
    v50 = (__m128i *)v39[8];
  }
  else
  {
    if ( v44 )
    {
      *v44 = v41;
      v45 = _mm_loadu_si128(&v88);
      v44[1] = v45;
      v44[2] = _mm_loadu_si128(&v89);
      if ( v87.m128i_i32[0] == 11 )
      {
        v44[2].m128i_i64[0] = 0;
        v46 = _mm_loadu_si128(&v88);
        v88 = v45;
        v44[1] = v46;
        v47 = v89.m128i_i64[0];
        v89.m128i_i64[0] = 0;
        v48 = v44[2].m128i_i64[1];
        v44[2].m128i_i64[0] = v47;
        v49 = v89.m128i_i64[1];
        v89.m128i_i64[1] = v48;
        v44[2].m128i_i64[1] = v49;
      }
      v44 = (__m128i *)v39[8];
    }
    v50 = v44 + 3;
    v39[8] = v44 + 3;
  }
  v51 = (__int64)v50->m128i_i64 - v39[7];
  if ( (unsigned __int64)v51 > 0x493E00 )
    goto LABEL_70;
  if ( v87.m128i_i32[0] == 11 && v89.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v89.m128i_i64[0])(&v88, &v88, 3);
  if ( v84.m128i_i32[0] == 11 && v86.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v86.m128i_i64[0])(&v85, &v85, 3);
  v81.m128i_i32[0] = 12;
  *(_QWORD *)(v19[7] + 48 * v38 + 8) = 0xAAAAAAAAAAAAAAABLL * (v51 >> 4) - 1;
  v52 = *(unsigned __int64 **)(a1 + 256);
  v81.m128i_i64[1] = -1;
  v53 = (__m128i *)v52[8];
  if ( v53 == (__m128i *)v52[9] )
  {
    sub_2BE00E0(v52 + 7, v53, &v81);
    v59 = v52[8];
  }
  else
  {
    if ( v53 )
    {
      *v53 = _mm_loadu_si128(&v81);
      v54 = _mm_loadu_si128(&v82);
      v28 = v81.m128i_i32[0] == 11;
      v53[1] = v54;
      v53[2] = _mm_loadu_si128(&v83);
      if ( v28 )
      {
        v55 = _mm_loadu_si128(&v82);
        v82 = v54;
        v56 = v53[2].m128i_i64[1];
        v53[2].m128i_i64[0] = 0;
        v53[1] = v55;
        v57 = v83.m128i_i64[0];
        v83.m128i_i64[0] = 0;
        v53[2].m128i_i64[0] = v57;
        v58 = v83.m128i_i64[1];
        v83.m128i_i64[1] = v56;
        v53[2].m128i_i64[1] = v58;
      }
      v53 = (__m128i *)v52[8];
    }
    v59 = (unsigned __int64)&v53[3];
    v52[8] = v59;
  }
  v60 = v59 - v52[7];
  if ( (unsigned __int64)v60 > 0x493E00 )
LABEL_70:
    abort();
  if ( v81.m128i_i32[0] == 11 && v83.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v83.m128i_i64[0])(&v82, &v82, 3);
  result = v19[7];
  *(_QWORD *)(result + v51 - 40) = 0xAAAAAAAAAAAAAAABLL * (v60 >> 4) - 1;
  v62 = *(_QWORD *)(a1 + 256);
  v63 = *(unsigned int **)(v62 + 64);
  for ( i = *(unsigned int **)(v62 + 56); v63 != i; i += 12 )
  {
    for ( j = *((_QWORD *)i + 1); j >= 0; *((_QWORD *)i + 1) = j )
    {
      v66 = *(_QWORD *)(v62 + 56) + 48 * j;
      if ( *(_DWORD *)v66 != 10 )
        break;
      j = *(_QWORD *)(v66 + 8);
    }
    result = *i;
    if ( (unsigned int)(result - 1) <= 1 || (_DWORD)result == 7 )
    {
      for ( result = *((_QWORD *)i + 2); result >= 0; *((_QWORD *)i + 2) = result )
      {
        result = *(_QWORD *)(v62 + 56) + 48 * result;
        if ( *(_DWORD *)result != 10 )
          break;
        result = *(_QWORD *)(result + 8);
      }
    }
  }
  return result;
}
