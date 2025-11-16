// Function: sub_330AD20
// Address: 0x330ad20
//
__int64 __fastcall sub_330AD20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v8; // rax
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // r13
  int v15; // edx
  int v16; // eax
  bool v17; // cl
  bool v18; // al
  int v19; // r9d
  unsigned __int16 *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int16 *v28; // rax
  __int64 v29; // r14
  int v30; // ebx
  unsigned __int16 *v31; // r13
  __m128i v32; // rax
  __int32 v33; // r10d
  bool v34; // zf
  __int32 v35; // r11d
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int32 v39; // edx
  __int128 v40; // rax
  int v41; // r9d
  __int64 v42; // rax
  __int32 v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // ecx
  __m128i v47; // xmm3
  __m128i si128; // xmm4
  __int64 v49; // rdi
  __int64 v50; // rax
  __int128 v51; // [rsp+0h] [rbp-D0h]
  __int64 v52; // [rsp+0h] [rbp-D0h]
  int v53; // [rsp+0h] [rbp-D0h]
  __int64 v54; // [rsp+18h] [rbp-B8h]
  __int64 v55; // [rsp+18h] [rbp-B8h]
  unsigned int v56; // [rsp+20h] [rbp-B0h]
  __m128i v57; // [rsp+20h] [rbp-B0h]
  __int32 v58; // [rsp+20h] [rbp-B0h]
  __m128i v59; // [rsp+30h] [rbp-A0h] BYREF
  __int128 v60; // [rsp+40h] [rbp-90h] BYREF
  __m128i v61; // [rsp+50h] [rbp-80h] BYREF
  __int64 v62; // [rsp+60h] [rbp-70h] BYREF
  int v63; // [rsp+68h] [rbp-68h]
  __m128i v64; // [rsp+70h] [rbp-60h] BYREF
  __m128i v65; // [rsp+80h] [rbp-50h]
  __m128i v66; // [rsp+90h] [rbp-40h]

  v8 = *(__int64 **)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *((_DWORD *)v8 + 2);
  v11 = *v8;
  v12 = v8[5];
  v62 = v9;
  v56 = v10;
  v13 = v8[10];
  v14 = *((unsigned int *)v8 + 22);
  v60 = (__int128)_mm_loadu_si128((const __m128i *)v8);
  v54 = v13;
  v59 = _mm_loadu_si128((const __m128i *)(v8 + 5));
  v61 = _mm_loadu_si128((const __m128i *)v8 + 5);
  if ( v9 )
    sub_B96E90((__int64)&v62, v9, 1);
  v15 = *(_DWORD *)(v11 + 24);
  v63 = *(_DWORD *)(a2 + 72);
  v16 = *(_DWORD *)(v12 + 24);
  v17 = v16 == 35;
  v18 = v16 == 11;
  if ( v15 != 35 && v15 != 11 || v17 || v18 )
  {
    if ( (unsigned __int8)sub_33CF170(v61.m128i_i64[0], v61.m128i_i64[1])
      && ((v20 = *(unsigned __int16 **)(a2 + 48), !*(_BYTE *)(a1 + 33))
       || ((v21 = *v20, v22 = *(_QWORD *)(a1 + 8), v23 = 1, (_WORD)v21 == 1)
        || (_WORD)v21 && (v23 = (unsigned __int16)v21, *(_QWORD *)(v22 + 8 * v21 + 112)))
       && (*(_BYTE *)(v22 + 500 * v23 + 6491) & 0xFB) == 0) )
    {
      v24 = sub_3411F20(
              *(_QWORD *)a1,
              77,
              (unsigned int)&v62,
              (_DWORD)v20,
              *(_DWORD *)(a2 + 68),
              v19,
              v60,
              *(_OWORD *)&v59);
    }
    else if ( (unsigned __int8)sub_33CF170(v60, *((_QWORD *)&v60 + 1))
           && (unsigned __int8)sub_33CF170(v59.m128i_i64[0], v59.m128i_i64[1]) )
    {
      v27 = *(_QWORD *)a1;
      v28 = (unsigned __int16 *)(*(_QWORD *)(v11 + 48) + 16LL * v56);
      v29 = *((_QWORD *)v28 + 1);
      v30 = *v28;
      v31 = (unsigned __int16 *)(*(_QWORD *)(v54 + 48) + 16 * v14);
      *((_QWORD *)&v51 + 1) = *((_QWORD *)v31 + 1);
      *(_QWORD *)&v51 = *v31;
      v60 = v51;
      v32.m128i_i64[0] = sub_33FB620(v27, v61.m128i_i32[0], v61.m128i_i32[2], (unsigned int)&v62, v30, v29, v51);
      v33 = v60;
      v34 = *(_DWORD *)(v32.m128i_i64[0] + 24) == 328;
      v61 = v32;
      v35 = DWORD2(v60);
      if ( v34 )
      {
        *(_QWORD *)&v60 = &v64;
      }
      else
      {
        v59.m128i_i64[0] = v32.m128i_i64[0];
        v57 = (__m128i)v60;
        v64.m128i_i64[0] = v61.m128i_i64[0];
        *(_QWORD *)&v60 = &v64;
        sub_32B3B20(a1 + 568, v64.m128i_i64);
        v36 = v59.m128i_i64[0];
        v33 = v57.m128i_i32[0];
        v35 = v57.m128i_i32[2];
        v37 = *(unsigned int *)(v59.m128i_i64[0] + 88);
        if ( (int)v37 < 0 )
        {
          *(_DWORD *)(v59.m128i_i64[0] + 88) = *(_DWORD *)(a1 + 48);
          v44 = *(unsigned int *)(a1 + 48);
          if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
          {
            sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v44 + 1, 8u, v36, v37);
            v44 = *(unsigned int *)(a1 + 48);
            v33 = v57.m128i_i32[0];
            v35 = v57.m128i_i32[2];
            v36 = v59.m128i_i64[0];
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v44) = v36;
          ++*(_DWORD *)(a1 + 48);
        }
      }
      v38 = sub_3400BD0(*(_QWORD *)a1, 0, (unsigned int)&v62, v33, v35, 0, 0);
      v58 = v39;
      LODWORD(v52) = 0;
      v59.m128i_i64[0] = *(_QWORD *)a1;
      v55 = v38;
      *(_QWORD *)&v40 = sub_3400BD0(v59.m128i_i32[0], 1, (unsigned int)&v62, v30, v29, 0, v52);
      v42 = sub_3406EB0(v59.m128i_i32[0], 186, (unsigned int)&v62, v30, v29, v41, *(_OWORD *)&v61, v40);
      v64.m128i_i32[2] = v43;
      v64.m128i_i64[0] = v42;
      v65.m128i_i64[0] = v55;
      v65.m128i_i32[2] = v58;
      v24 = sub_32EB790(a1, a2, (__int64 *)v60, 2, 1);
    }
    else
    {
      v24 = sub_3305B40(
              (__int64 *)a1,
              v60,
              *((__int64 *)&v60 + 1),
              v59.m128i_i64[0],
              v59.m128i_u64[1],
              a2,
              *(_OWORD *)&v61);
      if ( !v24 )
      {
        v26 = sub_3305B40(
                (__int64 *)a1,
                v59.m128i_i64[0],
                v59.m128i_i64[1],
                v60,
                *((unsigned __int64 *)&v60 + 1),
                a2,
                *(_OWORD *)&v61);
        if ( v26 )
        {
          v24 = v26;
        }
        else
        {
          v45 = *(_QWORD *)(a2 + 48);
          v46 = *(_DWORD *)(a2 + 68);
          v47 = _mm_load_si128(&v59);
          si128 = _mm_load_si128((const __m128i *)&v60);
          v53 = *(_DWORD *)(a2 + 28);
          v49 = *(_QWORD *)a1;
          v66 = _mm_load_si128(&v61);
          v64 = v47;
          v65 = si128;
          v50 = sub_33D00B0(v49, 72, v45, v46, (unsigned int)&v64, 3, v53);
          if ( v50 )
            v24 = v50;
        }
      }
    }
  }
  else
  {
    v24 = sub_3412970(
            *(_QWORD *)a1,
            72,
            (unsigned int)&v62,
            *(_QWORD *)(a2 + 48),
            *(_DWORD *)(a2 + 68),
            a6,
            *(_OWORD *)&v59,
            v60,
            *(_OWORD *)&v61);
  }
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  return v24;
}
