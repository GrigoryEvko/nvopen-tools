// Function: sub_68E150
// Address: 0x68e150
//
__int64 __fastcall sub_68E150(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // rdi
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 i; // r13
  _QWORD *v15; // r14
  __int64 v16; // rsi
  int v17; // edx
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __m128i v25; // xmm0
  __int8 v26; // al
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  unsigned int v42; // [rsp+2Ch] [rbp-1A4h] BYREF
  _QWORD v43[2]; // [rsp+30h] [rbp-1A0h] BYREF
  __m128i v44; // [rsp+40h] [rbp-190h] BYREF
  __m128i v45; // [rsp+50h] [rbp-180h] BYREF
  __m128i v46; // [rsp+60h] [rbp-170h] BYREF
  __m128i v47; // [rsp+70h] [rbp-160h] BYREF
  _BYTE v48[12]; // [rsp+80h] [rbp-150h] BYREF
  __int128 v49; // [rsp+8Ch] [rbp-144h] BYREF
  __m128i v50; // [rsp+A0h] [rbp-130h] BYREF
  __m128i v51; // [rsp+B0h] [rbp-120h] BYREF
  __m128i v52; // [rsp+C0h] [rbp-110h] BYREF
  __m128i v53; // [rsp+D0h] [rbp-100h] BYREF
  __m128i v54; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v55; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i v56; // [rsp+100h] [rbp-D0h] BYREF
  __m128i v57; // [rsp+110h] [rbp-C0h] BYREF
  __m128i v58; // [rsp+120h] [rbp-B0h] BYREF
  __m128i v59; // [rsp+130h] [rbp-A0h] BYREF
  __m128i v60; // [rsp+140h] [rbp-90h] BYREF
  __m128i v61; // [rsp+150h] [rbp-80h] BYREF
  __m128i v62; // [rsp+160h] [rbp-70h] BYREF
  __m128i v63; // [rsp+170h] [rbp-60h] BYREF
  __m128i v64; // [rsp+180h] [rbp-50h] BYREF
  __m128i v65[4]; // [rsp+190h] [rbp-40h] BYREF

  v7 = *(_QWORD *)a1;
  result = sub_8D3A70(v7);
  if ( (_DWORD)result )
  {
    v42 = 0;
    v13 = *(_QWORD *)a1;
    for ( i = *(_QWORD *)a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( a5 )
    {
      v15 = a5;
      do
      {
        v16 = v15[1];
        if ( v16 == v13 || (v7 = v13, (unsigned int)sub_8D97D0(v13, v16, 0, v10, v11)) )
        {
          if ( (unsigned int)sub_6E5430(v7, v16, v9, v10, v11, v12) )
            sub_685360(0x3F9u, (_DWORD *)(a1 + 68), i);
          return sub_6E6840(a1);
        }
        v15 = (_QWORD *)*v15;
      }
      while ( v15 );
      if ( (*(_BYTE *)(i + 177) & 0x20) == 0 )
        goto LABEL_20;
    }
    else
    {
      if ( (*(_BYTE *)(i + 177) & 0x20) != 0 )
        return result;
LABEL_20:
      sub_84EC30(41, 1, 1, 0, 1, a1, 0, a2, a3, a4, 0, (__int64)&v44, 0, 0, (__int64)&v42);
    }
    result = v42;
    if ( v42 )
    {
      v17 = *(_DWORD *)(a1 + 68);
      *(_WORD *)&v48[8] = *(_WORD *)(a1 + 72);
      *(_DWORD *)&v48[4] = v17;
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&v48[4];
      *(_QWORD *)&v49 = *(_QWORD *)(a1 + 76);
      unk_4F061D8 = v49;
      sub_6E3280(&v44, a2);
      v18 = _mm_loadu_si128(&v45);
      v19 = _mm_loadu_si128(&v46);
      v20 = _mm_loadu_si128(&v47);
      v21 = _mm_loadu_si128((const __m128i *)v48);
      v22 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 4));
      *(__m128i *)a1 = _mm_loadu_si128(&v44);
      v23 = _mm_loadu_si128(&v50);
      v24 = _mm_loadu_si128(&v51);
      *(__m128i *)(a1 + 16) = v18;
      v25 = _mm_loadu_si128(&v52);
      v26 = v45.m128i_i8[0];
      *(__m128i *)(a1 + 32) = v19;
      *(__m128i *)(a1 + 48) = v20;
      *(__m128i *)(a1 + 64) = v21;
      *(__m128i *)(a1 + 80) = v22;
      *(__m128i *)(a1 + 96) = v23;
      *(__m128i *)(a1 + 112) = v24;
      *(__m128i *)(a1 + 128) = v25;
      if ( v26 == 2 )
      {
        v27 = _mm_loadu_si128(&v54);
        v28 = _mm_loadu_si128(&v55);
        v29 = _mm_loadu_si128(&v56);
        v30 = _mm_loadu_si128(&v57);
        v31 = _mm_loadu_si128(&v58);
        *(__m128i *)(a1 + 144) = _mm_loadu_si128(&v53);
        v32 = _mm_loadu_si128(&v59);
        v33 = _mm_loadu_si128(&v60);
        *(__m128i *)(a1 + 160) = v27;
        *(__m128i *)(a1 + 176) = v28;
        v34 = _mm_loadu_si128(&v61);
        v35 = _mm_loadu_si128(&v62);
        *(__m128i *)(a1 + 192) = v29;
        v36 = _mm_loadu_si128(&v63);
        *(__m128i *)(a1 + 208) = v30;
        v37 = _mm_loadu_si128(&v64);
        *(__m128i *)(a1 + 224) = v31;
        v38 = _mm_loadu_si128(v65);
        *(__m128i *)(a1 + 240) = v32;
        *(__m128i *)(a1 + 256) = v33;
        *(__m128i *)(a1 + 272) = v34;
        *(__m128i *)(a1 + 288) = v35;
        *(__m128i *)(a1 + 304) = v36;
        *(__m128i *)(a1 + 320) = v37;
        *(__m128i *)(a1 + 336) = v38;
      }
      else if ( v26 == 5 || v26 == 1 )
      {
        *(_QWORD *)(a1 + 144) = v53.m128i_i64[0];
      }
      v43[0] = a5;
      v43[1] = v13;
      return sub_68E150(a1, a2, a3, a4 + 1, v43);
    }
  }
  return result;
}
