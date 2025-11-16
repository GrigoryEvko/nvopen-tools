// Function: sub_1FD99F0
// Address: 0x1fd99f0
//
__int64 __fastcall sub_1FD99F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  unsigned __int16 *v13; // rax
  int v14; // r8d
  __int32 v15; // edx
  unsigned __int16 *v16; // r9
  __int64 v17; // rax
  __int64 v18; // r12
  __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  __int64 *v22; // r15
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // r15d
  __int64 v28; // rax
  __int64 *v29; // r15
  __int64 v30; // r9
  __int64 v31; // r13
  __int64 v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  const __m128i *v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // r10
  __int64 *v39; // r13
  __int64 v40; // r15
  __int64 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 *v45; // [rsp+0h] [rbp-590h]
  unsigned __int8 v46; // [rsp+Fh] [rbp-581h]
  __int64 v47; // [rsp+18h] [rbp-578h]
  int v48; // [rsp+18h] [rbp-578h]
  __int64 v49; // [rsp+18h] [rbp-578h]
  _DWORD *v50; // [rsp+18h] [rbp-578h]
  __int64 v51; // [rsp+18h] [rbp-578h]
  unsigned __int16 *v52; // [rsp+18h] [rbp-578h]
  __m128i v53; // [rsp+20h] [rbp-570h] BYREF
  __m128i v54; // [rsp+30h] [rbp-560h] BYREF
  __int64 v55; // [rsp+40h] [rbp-550h]
  _DWORD *v56; // [rsp+50h] [rbp-540h] BYREF
  __int64 v57; // [rsp+58h] [rbp-538h]
  _DWORD v58[4]; // [rsp+60h] [rbp-530h] BYREF
  __int64 v59; // [rsp+70h] [rbp-520h]
  __int64 v60; // [rsp+78h] [rbp-518h]
  unsigned int v61; // [rsp+88h] [rbp-508h]
  __int64 v62; // [rsp+98h] [rbp-4F8h]
  __int64 v63; // [rsp+A0h] [rbp-4F0h]

  v56 = v58;
  v57 = 0x2000000000LL;
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 - 24 * v6);
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v8 = *(_QWORD *)(v7 + 24);
  else
    v8 = **(_QWORD **)(v7 + 24);
  v59 = 0;
  v60 = v8;
  LODWORD(v57) = 1;
  v58[0] = v58[0] & 0xFFF00000 | 1;
  v9 = *(_QWORD *)(a2 + 24 * (1 - v6));
  if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    v10 = *(_QWORD *)(v9 + 24);
  else
    v10 = **(_QWORD **)(v9 + 24);
  v63 = v10;
  v62 = 0;
  LODWORD(v57) = 2;
  v61 = v61 & 0xFFF00000 | 1;
  v46 = sub_1FD9530(a1, (__int64)&v56, a2, 2u, a5);
  if ( v46 )
  {
    v11 = a1[14];
    v12 = *(__int64 (**)())(*(_QWORD *)v11 + 1280LL);
    if ( v12 == sub_1FD3440 )
      BUG();
    v13 = (unsigned __int16 *)((__int64 (__fastcall *)(__int64, _QWORD))v12)(
                                v11,
                                (*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF);
    v15 = *v13;
    v16 = v13;
    if ( (_WORD)v15 )
    {
      v17 = (unsigned int)v57;
      LODWORD(v18) = 0;
      do
      {
        v54 = 0u;
        v53.m128i_i32[2] = v15;
        v55 = 0;
        v53.m128i_i64[0] = 0x430000000LL;
        if ( (unsigned int)v17 >= HIDWORD(v57) )
        {
          v52 = v16;
          sub_16CD150((__int64)&v56, v58, 0, 40, v14, (int)v16);
          v17 = (unsigned int)v57;
          v16 = v52;
        }
        v19 = (__m128i *)&v56[10 * v17];
        *v19 = _mm_loadu_si128(&v53);
        v19[1] = _mm_loadu_si128(&v54);
        v19[2].m128i_i64[0] = v55;
        v18 = (unsigned int)(v18 + 1);
        v17 = (unsigned int)(v57 + 1);
        LODWORD(v57) = v57 + 1;
        v15 = v16[v18];
      }
      while ( (_WORD)v15 );
    }
    v20 = a1[5];
    v45 = a1 + 10;
    v21 = *(_QWORD *)(v20 + 784);
    v22 = *(__int64 **)(v20 + 792);
    v23 = *(_QWORD *)(v21 + 56);
    v47 = v21;
    v24 = (__int64)sub_1E0B640(
                     v23,
                     *(_QWORD *)(a1[13] + 8LL) + ((unsigned __int64)*(unsigned int *)(a1[13] + 36LL) << 6),
                     a1 + 10,
                     0);
    sub_1DD5BA0((__int64 *)(v47 + 16), v24);
    v25 = *v22;
    v26 = *(_QWORD *)v24;
    *(_QWORD *)(v24 + 8) = v22;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v24 = v25 | v26 & 7;
    *(_QWORD *)(v25 + 8) = v24;
    *v22 = v24 | *v22 & 7;
    v48 = *(unsigned __int16 *)(*(_QWORD *)(v24 + 16) + 2LL);
    if ( *(_WORD *)(*(_QWORD *)(v24 + 16) + 2LL) )
    {
      v27 = 0;
      do
      {
        ++v27;
        v53.m128i_i64[0] = 1;
        v54 = 0u;
        sub_1E1A9C0(v24, v23, &v53);
      }
      while ( v27 != v48 );
    }
    v28 = a1[5];
    v29 = *(__int64 **)(v28 + 792);
    v30 = *(_QWORD *)(v28 + 784);
    v31 = *(_QWORD *)(v30 + 56);
    v49 = v30;
    v32 = (__int64)sub_1E0B640(v31, *(_QWORD *)(a1[13] + 8LL) + 1216LL, v45, 0);
    sub_1DD5BA0((__int64 *)(v49 + 16), v32);
    v33 = *v29;
    v34 = *(_QWORD *)v32;
    *(_QWORD *)(v32 + 8) = v29;
    v33 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v32 = v33 | v34 & 7;
    *(_QWORD *)(v33 + 8) = v32;
    *v29 = v32 | *v29 & 7;
    v35 = (unsigned __int64)v56;
    v50 = &v56[10 * (unsigned int)v57];
    if ( v56 != v50 )
    {
      do
      {
        v36 = (const __m128i *)v35;
        v35 += 40LL;
        sub_1E1A9C0(v32, v31, v36);
      }
      while ( v50 != (_DWORD *)v35 );
    }
    v37 = a1[5];
    v38 = *(_QWORD *)(v37 + 784);
    v39 = *(__int64 **)(v37 + 792);
    v40 = *(_QWORD *)(v38 + 56);
    v51 = v38;
    v41 = (__int64)sub_1E0B640(
                     v40,
                     *(_QWORD *)(a1[13] + 8LL) + ((unsigned __int64)*(unsigned int *)(a1[13] + 40LL) << 6),
                     v45,
                     0);
    sub_1DD5BA0((__int64 *)(v51 + 16), v41);
    v42 = *v39;
    v43 = *(_QWORD *)v41;
    *(_QWORD *)(v41 + 8) = v39;
    v42 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v41 = v42 | v43 & 7;
    *(_QWORD *)(v42 + 8) = v41;
    *v39 = v41 | *v39 & 7;
    v53.m128i_i64[0] = 1;
    v54 = 0u;
    sub_1E1A9C0(v41, v40, &v53);
    v53.m128i_i64[0] = 1;
    v54 = 0u;
    sub_1E1A9C0(v41, v40, &v53);
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1[5] + 8LL) + 56LL) + 39LL) = 1;
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  return v46;
}
