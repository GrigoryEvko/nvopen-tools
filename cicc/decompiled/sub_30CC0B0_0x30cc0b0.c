// Function: sub_30CC0B0
// Address: 0x30cc0b0
//
bool __fastcall sub_30CC0B0(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int128 a11,
        __int32 a12)
{
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rcx
  int v20; // r8d
  unsigned int v21; // eax
  void *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // rdi
  int v30; // r9d
  __m128i v31; // xmm4
  __m128i v32; // xmm3
  __m128i v33; // xmm2
  __m128i v34; // xmm1
  __m128i v35; // xmm0
  __int64 v36; // r15
  __m128i v37; // xmm3
  __m128i v38; // xmm7
  __m128i v39; // xmm0
  __m128i v40; // xmm1
  __m128i v41; // xmm2
  __int64 v42; // rax
  __m128i v43; // xmm4
  __m128i v44; // xmm7
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  __m128i v53; // xmm0
  __int64 v54; // rsi
  __m128i v55; // xmm1
  __m128i v56; // xmm2
  __m128i v57; // xmm3
  __m128i v58; // xmm4
  __m128i v59; // xmm0
  __m128i v60; // xmm1
  __m128i v61; // xmm2
  __int32 v62; // eax
  __int64 v63; // rdi
  _QWORD *v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rdx
  __m128i v68; // [rsp+0h] [rbp-1C0h]
  __m128i v69; // [rsp+10h] [rbp-1B0h]
  __m128i v70; // [rsp+20h] [rbp-1A0h]
  __m128i v71; // [rsp+30h] [rbp-190h]
  __m128i v72; // [rsp+40h] [rbp-180h]
  __int64 v73; // [rsp+58h] [rbp-168h]
  __int32 v74; // [rsp+6Ch] [rbp-154h]
  __m128i v75; // [rsp+70h] [rbp-150h] BYREF
  __m128i v76; // [rsp+80h] [rbp-140h] BYREF
  __m128i v77; // [rsp+90h] [rbp-130h] BYREF
  __m128i v78; // [rsp+A0h] [rbp-120h] BYREF
  __m128i v79; // [rsp+B0h] [rbp-110h] BYREF
  __int32 v80; // [rsp+C0h] [rbp-100h]
  _BYTE v81[24]; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v82; // [rsp+E8h] [rbp-D8h] BYREF
  __m128i v83; // [rsp+F8h] [rbp-C8h] BYREF
  __m128i v84; // [rsp+108h] [rbp-B8h] BYREF
  __m128i v85; // [rsp+118h] [rbp-A8h] BYREF
  __int32 v86; // [rsp+128h] [rbp-98h]
  __m128i v87; // [rsp+130h] [rbp-90h] BYREF
  __m128i v88; // [rsp+140h] [rbp-80h] BYREF
  __m128i v89; // [rsp+150h] [rbp-70h] BYREF
  __m128i v90; // [rsp+160h] [rbp-60h] BYREF
  __m128i v91; // [rsp+170h] [rbp-50h] BYREF
  __m128i v92[4]; // [rsp+180h] [rbp-40h] BYREF

  v15 = sub_BC0510(a1[1], &unk_4F82418, *a1);
  v16 = a1[1];
  v17 = *(_QWORD *)(v15 + 8);
  v18 = *(_DWORD *)(v16 + 24);
  v19 = *(_QWORD *)(v16 + 8);
  if ( !v18 )
    goto LABEL_8;
  v20 = v18 - 1;
  v21 = (v18 - 1) & (((unsigned int)&unk_502F108 >> 9) ^ ((unsigned int)&unk_502F108 >> 4));
  v22 = *(void **)(v19 + 16LL * v21);
  if ( v22 != &unk_502F108 )
  {
    v30 = 1;
    while ( v22 != (void *)-4096LL )
    {
      v21 = v20 & (v30 + v21);
      v22 = *(void **)(v19 + 16LL * v21);
      if ( v22 == &unk_502F108 )
        goto LABEL_3;
      ++v30;
    }
LABEL_8:
    v31 = _mm_loadu_si128((const __m128i *)&a7);
    v32 = _mm_loadu_si128((const __m128i *)&a8);
    v33 = _mm_loadu_si128((const __m128i *)&a9);
    v34 = _mm_loadu_si128((const __m128i *)&a10);
    v35 = _mm_loadu_si128((const __m128i *)&a11);
    *(__m128i *)&v81[8] = v31;
    v82 = v32;
    v86 = a12;
    v83 = v33;
    v84 = v34;
    v85 = v35;
    if ( a2 )
    {
      if ( a2 == 1 )
      {
        *(_QWORD *)v81 = v17;
        v37 = _mm_loadu_si128((const __m128i *)v81);
        v38 = _mm_loadu_si128((const __m128i *)&v82.m128i_u64[1]);
        v76.m128i_i64[0] = 0;
        v39 = _mm_loadu_si128((const __m128i *)&v83.m128i_u64[1]);
        v40 = _mm_loadu_si128((const __m128i *)&v84.m128i_u64[1]);
        v88 = _mm_loadu_si128((const __m128i *)&v81[16]);
        v41 = _mm_loadu_si128((const __m128i *)&v85.m128i_u64[1]);
        v87 = v37;
        v89 = v38;
        v90 = v39;
        v91 = v40;
        v92[0] = v41;
        v42 = sub_22077B0(0x60u);
        if ( v42 )
        {
          v43 = _mm_loadu_si128(&v88);
          v44 = _mm_loadu_si128(&v89);
          v45 = _mm_loadu_si128(&v90);
          v46 = _mm_loadu_si128(&v91);
          v47 = _mm_loadu_si128(v92);
          *(__m128i *)v42 = _mm_loadu_si128(&v87);
          *(__m128i *)(v42 + 16) = v43;
          *(__m128i *)(v42 + 32) = v44;
          *(__m128i *)(v42 + 48) = v45;
          *(__m128i *)(v42 + 64) = v46;
          *(__m128i *)(v42 + 80) = v47;
        }
        v48 = a1[1];
        v75.m128i_i64[0] = v42;
        v49 = *a1;
        v76.m128i_i64[1] = (__int64)sub_30D0E60;
        v76.m128i_i64[0] = (__int64)sub_30CA430;
        sub_30FFC20(&v87, v49, v48, &v75);
        v50 = v87.m128i_i64[0];
        v51 = a1[2];
        v87.m128i_i64[0] = 0;
        a1[2] = v50;
        if ( v51 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v51 + 8LL))(v51);
          if ( v87.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v87.m128i_i64[0] + 8LL))(v87.m128i_i64[0]);
        }
        if ( v76.m128i_i64[0] )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v76.m128i_i64[0])(&v75, &v75, 3);
      }
    }
    else
    {
      v80 = a12;
      v75 = v31;
      v76 = v32;
      v77 = v33;
      v78 = v34;
      v79 = v35;
      v52 = sub_22077B0(0xA8u);
      v36 = v52;
      if ( v52 )
      {
        v53 = _mm_loadu_si128(&v76);
        LOBYTE(v74) = 1;
        v54 = *a1;
        v55 = _mm_loadu_si128(&v77);
        v56 = _mm_loadu_si128(&v78);
        v87 = _mm_loadu_si128(&v75);
        v57 = _mm_loadu_si128(&v79);
        v92[0].m128i_i32[0] = v80;
        v88 = v53;
        v89 = v55;
        v90 = v56;
        v91 = v57;
        sub_30CBEF0(v52, v54, v17, a4, v74);
        v58 = _mm_loadu_si128(&v87);
        v59 = _mm_loadu_si128(&v89);
        v60 = _mm_loadu_si128(&v90);
        v61 = _mm_loadu_si128(&v91);
        *(__m128i *)(v36 + 96) = _mm_loadu_si128(&v88);
        *(_QWORD *)v36 = &unk_4A32558;
        v62 = v92[0].m128i_i32[0];
        *(__m128i *)(v36 + 80) = v58;
        *(_DWORD *)(v36 + 160) = v62;
        *(__m128i *)(v36 + 112) = v59;
        *(__m128i *)(v36 + 128) = v60;
        *(__m128i *)(v36 + 144) = v61;
        sub_30CA8B0(v36);
      }
      v63 = a1[2];
      a1[2] = v36;
      if ( v63 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v63 + 8LL))(v63);
        v36 = a1[2];
      }
      if ( !*(_QWORD *)(a3 + 8) )
        return v36 != 0;
      v64 = (_QWORD *)*a1;
      a1[2] = 0;
      v75.m128i_i64[0] = v36;
      sub_310A360((unsigned int)&v87, (_DWORD)v64, v17, *v64, (unsigned int)&v75, a3, 1, a4);
      v65 = v87.m128i_i64[0];
      v66 = a1[2];
      v87.m128i_i64[0] = 0;
      a1[2] = v65;
      v67 = v73;
      if ( v66 )
      {
        (*(void (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v66 + 8LL))(v66, v64, v73);
        if ( v87.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v87.m128i_i64[0] + 8LL))(v87.m128i_i64[0]);
      }
      if ( v75.m128i_i64[0] )
        (*(void (__fastcall **)(__int64, _QWORD *, __int64))(*(_QWORD *)v75.m128i_i64[0] + 8LL))(
          v75.m128i_i64[0],
          v64,
          v67);
    }
    v36 = a1[2];
    return v36 != 0;
  }
LABEL_3:
  v23 = sub_BC0510(v16, &unk_502F108, *a1);
  v68 = _mm_loadu_si128((const __m128i *)&a7);
  v69 = _mm_loadu_si128((const __m128i *)&a8);
  v70 = _mm_loadu_si128((const __m128i *)&a9);
  v71 = _mm_loadu_si128((const __m128i *)&a10);
  v72 = _mm_loadu_si128((const __m128i *)&a11);
  v27 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int32))(v23 + 8))(
          *a1,
          v17,
          a4,
          v24,
          v25,
          v26,
          v68.m128i_i64[0],
          v68.m128i_i64[1],
          v69.m128i_i64[0],
          v69.m128i_i64[1],
          v70.m128i_i64[0],
          v70.m128i_i64[1],
          v71.m128i_i64[0],
          v71.m128i_i64[1],
          v72.m128i_i64[0],
          v72.m128i_i64[1],
          a12);
  v28 = a1[2];
  a1[2] = v27;
  if ( v28 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 8LL))(v28);
    v27 = a1[2];
  }
  return v27 != 0;
}
