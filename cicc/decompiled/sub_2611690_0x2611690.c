// Function: sub_2611690
// Address: 0x2611690
//
__int64 __fastcall sub_2611690(_DWORD *a1, void *a2, __int64 a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int i; // eax
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned int v14; // r15d
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm4
  __m128i v27; // xmm6
  __int32 v28; // edx
  __m128i v29; // xmm5
  __int64 v30; // rdi
  __int64 v31; // r14
  __m128i *v32; // rax
  __int64 v33; // rcx
  __int32 v34; // edx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // [rsp-1B0h] [rbp-1B0h]
  __int64 v40; // [rsp-1B0h] [rbp-1B0h]
  __m128i *v41; // [rsp-1A0h] [rbp-1A0h]
  unsigned int v42; // [rsp-18Ch] [rbp-18Ch]
  __m128i v43; // [rsp-188h] [rbp-188h] BYREF
  __m128i v44; // [rsp-178h] [rbp-178h] BYREF
  __m128i v45; // [rsp-168h] [rbp-168h] BYREF
  __m128i v46; // [rsp-158h] [rbp-158h] BYREF
  __m128i v47; // [rsp-148h] [rbp-148h] BYREF
  int v48; // [rsp-138h] [rbp-138h]
  __m128i v49; // [rsp-128h] [rbp-128h] BYREF
  __m128i v50; // [rsp-118h] [rbp-118h] BYREF
  __m128i v51; // [rsp-108h] [rbp-108h] BYREF
  __m128i v52; // [rsp-F8h] [rbp-F8h] BYREF
  __m128i v53; // [rsp-E8h] [rbp-E8h] BYREF
  int v54; // [rsp-D8h] [rbp-D8h]
  __int128 v55; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v56; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v57; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v58; // [rsp-98h] [rbp-98h] BYREF
  __m128i v59; // [rsp-88h] [rbp-88h] BYREF
  __int32 v60; // [rsp-78h] [rbp-78h]
  __int64 v61; // [rsp-38h] [rbp-38h] BYREF
  __int64 v62; // [rsp-8h] [rbp-8h] BYREF

  result = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    v8 = *(unsigned int *)(*(_QWORD *)a2 + 88LL);
    v9 = *(_QWORD *)(*(_QWORD *)a2 + 72LL);
    if ( !(_DWORD)v8 )
      goto LABEL_8;
    a2 = &unk_502F110;
    v10 = 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_502F110 >> 9) ^ ((unsigned int)&unk_502F110 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)))); ; i = (v8 - 1) & v13 )
    {
      v12 = v9 + 24LL * i;
      if ( *(_UNKNOWN **)v12 == &unk_502F110 && a4 == *(__int64 **)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
        goto LABEL_8;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != v9 + 24 * v8 && (v31 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL)) != 0 )
    {
      *((_QWORD *)&v55 + 1) = 1;
      v32 = &v56;
      do
      {
        v32->m128i_i64[0] = -4096;
        ++v32;
      }
      while ( v32 != (__m128i *)&v61 );
      if ( (BYTE8(v55) & 1) == 0 )
        sub_C7D6A0(v56.m128i_i64[0], 16LL * v56.m128i_u32[2], 8);
      return *(_QWORD *)(v31 + 24);
    }
    else
    {
LABEL_8:
      v14 = a1[3];
      sub_30D6B30(&v43, a2, v8, v9);
      v15 = _mm_loadu_si128(&v44);
      v16 = _mm_loadu_si128(&v45);
      v17 = _mm_loadu_si128(&v46);
      v18 = _mm_loadu_si128(&v47);
      v49 = _mm_loadu_si128(&v43);
      v50 = v15;
      v54 = v48;
      v51 = v16;
      v52 = v17;
      v53 = v18;
      result = sub_22077B0(0xA8u);
      if ( result )
      {
        v19 = _mm_loadu_si128(&v49);
        v20 = _mm_loadu_si128(&v50);
        v21 = _mm_loadu_si128(&v51);
        v22 = _mm_loadu_si128(&v52);
        v23 = _mm_loadu_si128(&v53);
        v60 = v54;
        LOBYTE(v42) = 1;
        v41 = (__m128i *)result;
        v55 = (__int128)v19;
        v56 = v20;
        v57 = v21;
        v58 = v22;
        v59 = v23;
        sub_30CBEF0(result, a4, a3, v14 | 0x100000000LL, v42);
        v24 = _mm_loadu_si128((const __m128i *)&v55);
        v25 = _mm_loadu_si128(&v56);
        v26 = _mm_loadu_si128(&v57);
        v27 = _mm_loadu_si128(&v59);
        v41->m128i_i64[0] = (__int64)&unk_4A32558;
        v28 = v60;
        v29 = _mm_loadu_si128(&v58);
        v41[5] = v24;
        v41[10].m128i_i32[0] = v28;
        v41[6] = v25;
        v41[7] = v26;
        v41[8] = v29;
        v41[9] = v27;
        sub_30CA8B0(v41);
        result = (__int64)v41;
      }
      v30 = *(_QWORD *)a1;
      *(_QWORD *)a1 = result;
      if ( v30 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
        result = *(_QWORD *)a1;
      }
      if ( *((_QWORD *)&xmmword_4FF2748 + 1) )
      {
        v33 = (unsigned int)a1[3];
        v49.m128i_i64[0] = result;
        v56.m128i_i64[0] = __PAIR64__(dword_4FF2288, dword_4FF24E8);
        v55 = xmmword_4FF2748;
        v34 = dword_4FF2028;
        *(_QWORD *)a1 = 0;
        v39 = v33 | 0x500000000LL;
        v35 = *a4;
        v56.m128i_i32[2] = v34;
        sub_310A360((unsigned int)&v62 - 384, (_DWORD)a4, a3, v35, (unsigned int)&v49, (unsigned int)&v55, 1, v39);
        v36 = v43.m128i_i64[0];
        v37 = *(_QWORD *)a1;
        v43.m128i_i64[0] = 0;
        *(_QWORD *)a1 = v36;
        v38 = v40;
        if ( v37 )
        {
          (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v37 + 8LL))(v37, a4, v40);
          if ( v43.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v43.m128i_i64[0] + 8LL))(v43.m128i_i64[0]);
        }
        if ( v49.m128i_i64[0] )
          (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v49.m128i_i64[0] + 8LL))(
            v49.m128i_i64[0],
            a4,
            v38);
        return *(_QWORD *)a1;
      }
    }
  }
  return result;
}
