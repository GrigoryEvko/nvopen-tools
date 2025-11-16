// Function: sub_21459C0
// Address: 0x21459c0
//
__int64 __fastcall sub_21459C0(__int64 a1, __int64 a2, __m128i a3, __m128 a4, __m128i a5)
{
  char *v7; // rdx
  char v8; // al
  const void **v9; // rdx
  __int64 v10; // rbx
  unsigned __int8 *v11; // rax
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  bool v19; // al
  __int64 v20; // rdx
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // ecx
  __int64 v26; // rax
  unsigned __int64 *v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rax
  bool v33; // al
  _QWORD *v34; // r13
  unsigned __int8 v35; // al
  const void **v36; // r8
  __int128 v37; // rax
  __int64 v38; // r12
  const void **v40; // rdx
  __int128 v41; // [rsp-10h] [rbp-200h]
  __int64 v42; // [rsp+8h] [rbp-1E8h]
  unsigned __int8 v43; // [rsp+17h] [rbp-1D9h]
  __int64 v44; // [rsp+38h] [rbp-1B8h]
  __int64 v45; // [rsp+48h] [rbp-1A8h]
  unsigned int v46; // [rsp+48h] [rbp-1A8h]
  unsigned int v47; // [rsp+60h] [rbp-190h] BYREF
  const void **v48; // [rsp+68h] [rbp-188h]
  __int64 v49; // [rsp+70h] [rbp-180h] BYREF
  int v50; // [rsp+78h] [rbp-178h]
  __m128i v51; // [rsp+80h] [rbp-170h] BYREF
  __m128i v52; // [rsp+90h] [rbp-160h] BYREF
  _BYTE v53[8]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v54; // [rsp+A8h] [rbp-148h]
  _QWORD *v55; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-138h]
  _QWORD v57[38]; // [rsp+C0h] [rbp-130h] BYREF

  v7 = *(char **)(a2 + 40);
  v8 = *v7;
  v9 = (const void **)*((_QWORD *)v7 + 1);
  LOBYTE(v47) = v8;
  v48 = v9;
  if ( v8 )
    v10 = word_4310E40[(unsigned __int8)(v8 - 14)];
  else
    v10 = (unsigned int)sub_1F58D30((__int64)&v47);
  v11 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  sub_1F40D10((__int64)&v55, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v11, *((_QWORD *)v11 + 1));
  v14 = *(_QWORD *)(a2 + 72);
  v43 = v56;
  v49 = v14;
  v42 = v57[0];
  if ( v14 )
    sub_1623A60((__int64)&v49, v14, 2);
  v15 = (unsigned int)(2 * v10);
  v50 = *(_DWORD *)(a2 + 64);
  v55 = v57;
  v56 = 0x1000000000LL;
  if ( v15 > 0x10 )
    sub_16CD150((__int64)&v55, v57, v15, 16, v12, v13);
  if ( (_DWORD)v10 )
  {
    v16 = 5 * v10;
    v17 = 0;
    v45 = 8 * v16;
    while ( 1 )
    {
      v51.m128i_i32[2] = 0;
      v26 = *(_QWORD *)(a2 + 32);
      v52.m128i_i32[2] = 0;
      v51.m128i_i64[0] = 0;
      v27 = (unsigned __int64 *)(v17 + v26);
      v52.m128i_i64[0] = 0;
      v28 = v27[1];
      v29 = *v27;
      v30 = *(_QWORD *)(*v27 + 40) + 16LL * *((unsigned int *)v27 + 2);
      v31 = *(_BYTE *)v30;
      v32 = *(_QWORD *)(v30 + 8);
      v53[0] = v31;
      v54 = v32;
      if ( v31 )
      {
        v18 = a1;
        v19 = (unsigned __int8)(v31 - 14) <= 0x47u || (unsigned __int8)(v31 - 2) <= 5u;
        v20 = v28;
        if ( !v19 )
          goto LABEL_20;
      }
      else
      {
        v44 = v28;
        v33 = sub_1F58CF0((__int64)v53);
        v18 = a1;
        v20 = v44;
        if ( !v33 )
        {
LABEL_20:
          sub_2016B80(v18, v29, v20, &v51, &v52);
          goto LABEL_11;
        }
      }
      sub_20174B0(v18, v29, v20, &v51, &v52);
LABEL_11:
      if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
      {
        a3 = _mm_load_si128(&v51);
        v51.m128i_i64[0] = v52.m128i_i64[0];
        v51.m128i_i32[2] = v52.m128i_i32[2];
        v52.m128i_i64[0] = a3.m128i_i64[0];
        v52.m128i_i32[2] = a3.m128i_i32[2];
      }
      v23 = (unsigned int)v56;
      if ( (unsigned int)v56 >= HIDWORD(v56) )
      {
        sub_16CD150((__int64)&v55, v57, 0, 16, v21, v22);
        v23 = (unsigned int)v56;
      }
      a4 = (__m128)_mm_load_si128(&v51);
      *(__m128 *)&v55[2 * v23] = a4;
      v24 = (unsigned int)(v56 + 1);
      LODWORD(v56) = v24;
      if ( HIDWORD(v56) <= (unsigned int)v24 )
      {
        sub_16CD150((__int64)&v55, v57, 0, 16, v21, v22);
        v24 = (unsigned int)v56;
      }
      a5 = _mm_load_si128(&v52);
      v17 += 40;
      *(__m128i *)&v55[2 * v24] = a5;
      v25 = v56 + 1;
      LODWORD(v56) = v56 + 1;
      if ( v45 == v17 )
        goto LABEL_22;
    }
  }
  v25 = v56;
LABEL_22:
  v46 = v25;
  v34 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  v35 = sub_1D15020(v43, v25);
  v36 = 0;
  if ( !v35 )
  {
    v35 = sub_1F593D0(v34, v43, v42, v46);
    v36 = v40;
  }
  *((_QWORD *)&v41 + 1) = (unsigned int)v56;
  *(_QWORD *)&v41 = v55;
  *(_QWORD *)&v37 = sub_1D359D0(
                      *(__int64 **)(a1 + 8),
                      104,
                      (__int64)&v49,
                      v35,
                      v36,
                      0,
                      *(double *)a3.m128i_i64,
                      *(double *)a4.m128_u64,
                      a5,
                      v41);
  v38 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          158,
          (__int64)&v49,
          v47,
          v48,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128_u64,
          *(double *)a5.m128i_i64,
          v37);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( v49 )
    sub_161E7C0((__int64)&v49, v49);
  return v38;
}
