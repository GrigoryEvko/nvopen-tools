// Function: sub_33AC4A0
// Address: 0x33ac4a0
//
void __fastcall sub_33AC4A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        char a9)
{
  __int64 m128i_i64; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i *v16; // rsi
  int v17; // edx
  __int32 v18; // edx
  __int32 v19; // eax
  __m128i *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rsi
  int v24; // edx
  unsigned __int16 v25; // r12
  char v26; // al
  __m128i v27; // xmm3
  const __m128i *v28; // rcx
  unsigned __int64 v29; // rdi
  char v30; // dl
  char v31; // al
  char v32; // al
  __m128i *v33; // rdx
  __int64 v34; // rax
  const __m128i *v35; // rax
  bool v36; // dl
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // r13
  __int64 v41; // r12
  int v42; // r14d
  __int64 v43; // r13
  __int64 v44; // rax
  bool v45; // al
  __int64 v46; // rsi
  char v48; // [rsp+10h] [rbp-D0h]
  unsigned int i; // [rsp+14h] [rbp-CCh]
  __int64 v50; // [rsp+18h] [rbp-C8h]
  __int64 v51; // [rsp+58h] [rbp-88h] BYREF
  const __m128i *v52; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v53; // [rsp+68h] [rbp-78h]
  const __m128i *v54; // [rsp+70h] [rbp-70h]
  __m128i v55; // [rsp+80h] [rbp-60h] BYREF
  __m128i v56; // [rsp+90h] [rbp-50h] BYREF
  __m128i v57[4]; // [rsp+A0h] [rbp-40h] BYREF

  v52 = 0;
  v48 = a9;
  v53 = 0;
  v54 = 0;
  sub_3375F60(&v52, a5);
  for ( i = a5 + a4; i != a4; v53 = v16 + 3 )
  {
    while ( 1 )
    {
      v17 = *(_DWORD *)(a3 + 4);
      v55 = 0u;
      v56 = 0u;
      v57[0] = 0u;
      v50 = *(_QWORD *)(a3 + 32 * (a4 - (unsigned __int64)(v17 & 0x7FFFFFF)));
      v55.m128i_i64[1] = sub_338B750(a1, v50);
      v56.m128i_i32[0] = v18;
      v56.m128i_i64[1] = *(_QWORD *)(v50 + 8);
      sub_34470B0(&v55, a3, a4);
      v16 = v53;
      if ( v53 != v54 )
        break;
      ++a4;
      sub_332CDC0((unsigned __int64 *)&v52, v53, &v55);
      if ( i == a4 )
        goto LABEL_8;
    }
    if ( v53 )
    {
      *v53 = _mm_loadu_si128(&v55);
      v16[1] = _mm_loadu_si128(&v56);
      v16[2] = _mm_loadu_si128(v57);
      v16 = v53;
    }
    ++a4;
  }
LABEL_8:
  v19 = *(_DWORD *)(a1 + 848);
  v20 = *(__m128i **)a1;
  v55.m128i_i64[0] = 0;
  v55.m128i_i32[2] = v19;
  if ( !v20
    || (m128i_i64 = (__int64)v20[3].m128i_i64, &v55 == &v20[3])
    || (v21 = v20[3].m128i_i64[0], (v55.m128i_i64[0] = v21) == 0) )
  {
    v23 = *(_QWORD *)(a2 + 88);
    if ( !v23 )
    {
      v22 = v55.m128i_i64[0];
      *(_QWORD *)(a2 + 88) = v55.m128i_i64[0];
      goto LABEL_16;
    }
    goto LABEL_12;
  }
  sub_B96E90((__int64)&v55, v21, 1);
  v23 = *(_QWORD *)(a2 + 88);
  if ( v23 )
LABEL_12:
    sub_B91220(a2 + 88, v23);
  v23 = v55.m128i_i64[0];
  *(_QWORD *)(a2 + 88) = v55.m128i_i64[0];
  if ( v23 )
    sub_B96E90(a2 + 88, v23, 1);
  v19 = v55.m128i_i32[2];
LABEL_16:
  *(_DWORD *)(a2 + 96) = v19;
  *(_QWORD *)a2 = sub_33738B0(a1, v23, v22, m128i_i64, v14, v15);
  *(_DWORD *)(a2 + 8) = v24;
  v25 = *(_WORD *)(a3 + 2);
  v51 = a8;
  *(_QWORD *)(a2 + 16) = a6;
  *(_BYTE *)(a2 + 24) = (8 * (sub_A73170(&v51, 15) & 1)) | *(_BYTE *)(a2 + 24) & 0xF7;
  *(_BYTE *)(a2 + 24) = sub_A73170(&v51, 54) & 1 | *(_BYTE *)(a2 + 24) & 0xFE;
  *(_BYTE *)(a2 + 24) = (2 * (sub_A73170(&v51, 79) & 1)) | *(_BYTE *)(a2 + 24) & 0xFD;
  v26 = sub_A73170(&v51, 32);
  v27 = _mm_loadu_si128((const __m128i *)&a7);
  v28 = v52;
  v29 = *(_QWORD *)(a2 + 56);
  *(_DWORD *)(a2 + 32) = (v25 >> 2) & 0x3FF;
  v30 = 2 * (v26 & 1);
  v31 = *(_BYTE *)(a2 + 25);
  *(_QWORD *)(a2 + 56) = v28;
  v52 = 0;
  v32 = v30 | v31 & 0xFD;
  v33 = v53;
  v53 = 0;
  *(_BYTE *)(a2 + 25) = v32;
  v34 = a7;
  *(_QWORD *)(a2 + 64) = v33;
  *(_QWORD *)(a2 + 40) = v34;
  *(_DWORD *)(a2 + 48) = v27.m128i_i32[2];
  *(_DWORD *)(a2 + 28) = -1431655765 * (v33 - v28);
  v35 = v54;
  v54 = 0;
  *(_QWORD *)(a2 + 72) = v35;
  if ( v29 )
    j_j___libc_free_0(v29);
  v36 = 0;
  *(_BYTE *)(a2 + 24) = *(_BYTE *)(a2 + 24) & 0x5F | (v48 << 7) | (32 * (*(_QWORD *)(a3 + 16) != 0));
  if ( *(char *)(a3 + 7) < 0 )
  {
    v37 = sub_BD2BC0(a3);
    v39 = v37 + v38;
    if ( *(char *)(a3 + 7) < 0 )
      v39 -= sub_BD2BC0(a3);
    v40 = v39 >> 4;
    if ( (_DWORD)v40 )
    {
      v41 = 0;
      v42 = 0;
      v43 = 16LL * (unsigned int)v40;
      do
      {
        v44 = 0;
        if ( *(char *)(a3 + 7) < 0 )
          v44 = sub_BD2BC0(a3);
        v45 = *(_DWORD *)(*(_QWORD *)(v44 + v41) + 8LL) == 4;
        v41 += 16;
        v42 += v45;
      }
      while ( v41 != v43 );
      v36 = v42 != 0;
    }
    else
    {
      v36 = 0;
    }
  }
  v46 = v55.m128i_i64[0];
  *(_BYTE *)(a2 + 25) = v36 | *(_BYTE *)(a2 + 25) & 0xFE;
  if ( v46 )
    sub_B91220((__int64)&v55, v46);
  if ( v52 )
    j_j___libc_free_0((unsigned __int64)v52);
}
