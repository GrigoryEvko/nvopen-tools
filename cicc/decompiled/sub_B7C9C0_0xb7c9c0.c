// Function: sub_B7C9C0
// Address: 0xb7c9c0
//
__int64 __fastcall sub_B7C9C0(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  int v7; // r14d
  int v8; // r15d
  __int32 v11; // eax
  int v12; // edx
  __int64 v13; // rax
  const char *v14; // rdx
  size_t v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rdx
  __m128i v25; // xmm0
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 i; // rbx
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdi
  int v32; // eax
  __m128i *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  char v40; // cl
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  unsigned __int64 v45; // r10
  const __m128i *v46; // rax
  __m128i *v47; // rcx
  __int64 v48; // [rsp+8h] [rbp-B8h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  int v50; // [rsp+18h] [rbp-A8h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  int v52; // [rsp+20h] [rbp-A0h]
  __int64 v53; // [rsp+20h] [rbp-A0h]
  __int64 v54; // [rsp+20h] [rbp-A0h]
  __int64 v55; // [rsp+28h] [rbp-98h]
  char *v56; // [rsp+28h] [rbp-98h]
  __int64 v57; // [rsp+28h] [rbp-98h]
  __int64 v58; // [rsp+30h] [rbp-90h]
  __int64 v59; // [rsp+30h] [rbp-90h]
  __int64 v60; // [rsp+30h] [rbp-90h]
  const char *v61; // [rsp+38h] [rbp-88h]
  __int64 v62; // [rsp+38h] [rbp-88h]
  __m128i v63; // [rsp+50h] [rbp-70h] BYREF

  *(_QWORD *)(a1 + 104) = a1 + 120;
  v48 = a1 + 120;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = 0x500000000LL;
  v11 = a3->m128i_i32[2];
  v12 = 0;
  v13 = (unsigned int)(v11 - 13);
  if ( (unsigned int)v13 <= 8 )
    v12 = dword_3F54D20[v13];
  *(_DWORD *)a1 = v12;
  v14 = (const char *)a3[2].m128i_i64[1];
  v15 = 0;
  if ( v14 )
  {
    v49 = a6;
    v58 = a5;
    v61 = (const char *)a3[2].m128i_i64[1];
    v15 = strlen(v61);
    a6 = v49;
    a5 = v58;
    v14 = v61;
  }
  *(_QWORD *)(a1 + 8) = v14;
  *(_QWORD *)(a1 + 16) = v15;
  v16 = a3[3].m128i_i64[1];
  v59 = a6;
  *(_QWORD *)(a1 + 24) = a3[3].m128i_i64[0];
  *(_QWORD *)(a1 + 32) = v16;
  v62 = a5;
  v17 = (_BYTE *)sub_BD5D20(a3[1].m128i_i64[0]);
  v19 = v62;
  v20 = v59;
  if ( v18 && *v17 == 1 )
  {
    --v18;
    ++v17;
  }
  *(_QWORD *)(a1 + 40) = v17;
  *(_QWORD *)(a1 + 48) = v18;
  v21 = a3[1].m128i_i64[1];
  v63.m128i_i64[1] = a3[2].m128i_i64[0];
  v22 = 0;
  v63.m128i_i64[0] = v21;
  if ( v21 )
  {
    v23 = sub_B159E0(v63.m128i_i64, a2);
    v20 = v59;
    v19 = v62;
    v55 = v24;
    v6 = v23;
    v22 = 1;
    v50 = v63.m128i_i32[3];
    v52 = v63.m128i_i32[2];
  }
  *(_QWORD *)(a1 + 56) = v6;
  *(_BYTE *)(a1 + 80) = v22;
  *(_QWORD *)(a1 + 64) = v55;
  *(_DWORD *)(a1 + 72) = v52;
  *(_DWORD *)(a1 + 76) = v50;
  v25 = _mm_loadu_si128(a3 + 4);
  *(__m128i *)(a1 + 88) = v25;
  v26 = a3[5].m128i_i64[0];
  v27 = 5LL * a3[5].m128i_u32[2];
  v63 = v25;
  v60 = a1 + 104;
  for ( i = v26 + 16 * v27; i != v26; *(_BYTE *)(v43 + 56) = v40 )
  {
    v29 = *(unsigned int *)(a1 + 112);
    v30 = *(_QWORD *)(a1 + 104);
    v31 = *(unsigned int *)(a1 + 116);
    v32 = *(_DWORD *)(a1 + 112);
    v33 = (__m128i *)(v30 + (v29 << 6));
    if ( v29 >= v31 )
    {
      v45 = v29 + 1;
      v46 = &v63;
      memset(&v63, 0, 64);
      if ( v31 < v29 + 1 )
      {
        if ( v30 > (unsigned __int64)&v63 || v33 <= &v63 )
        {
          v54 = v20;
          v57 = v19;
          sub_C8D5F0(v60, v48, v45, 64);
          v30 = *(_QWORD *)(a1 + 104);
          v29 = *(unsigned int *)(a1 + 112);
          v46 = &v63;
          v20 = v54;
          v19 = v57;
        }
        else
        {
          v51 = v20;
          v53 = v19;
          v56 = &v63.m128i_i8[-v30];
          sub_C8D5F0(v60, v48, v45, 64);
          v30 = *(_QWORD *)(a1 + 104);
          v29 = *(unsigned int *)(a1 + 112);
          v19 = v53;
          v20 = v51;
          v46 = (const __m128i *)&v56[v30];
        }
      }
      v47 = (__m128i *)((v29 << 6) + v30);
      *v47 = _mm_loadu_si128(v46);
      v47[1] = _mm_loadu_si128(v46 + 1);
      v47[2] = _mm_loadu_si128(v46 + 2);
      v47[3] = _mm_loadu_si128(v46 + 3);
      v30 = *(_QWORD *)(a1 + 104);
      v34 = (unsigned int)(*(_DWORD *)(a1 + 112) + 1);
      *(_DWORD *)(a1 + 112) = v34;
    }
    else
    {
      if ( v33 )
      {
        *v33 = 0;
        v33[1] = 0;
        v33[2] = 0;
        v33[3] = 0;
        v32 = *(_DWORD *)(a1 + 112);
        v30 = *(_QWORD *)(a1 + 104);
      }
      v34 = (unsigned int)(v32 + 1);
      *(_DWORD *)(a1 + 112) = v34;
    }
    v35 = *(_QWORD *)(v26 + 8);
    v36 = (_QWORD *)(v30 + (v34 << 6) - 64);
    *v36 = *(_QWORD *)v26;
    v36[1] = v35;
    v37 = *(_QWORD *)(a1 + 104);
    v38 = *(_QWORD *)(v26 + 40);
    v39 = v37 + ((unsigned __int64)*(unsigned int *)(a1 + 112) << 6) - 64;
    *(_QWORD *)(v39 + 16) = *(_QWORD *)(v26 + 32);
    v40 = 0;
    *(_QWORD *)(v39 + 24) = v38;
    if ( *(_QWORD *)(v26 + 64) )
    {
      v41 = sub_B159E0((__int64 *)(v26 + 64), v37);
      v7 = *(_DWORD *)(v26 + 72);
      v8 = *(_DWORD *)(v26 + 76);
      v40 = 1;
      v19 = v41;
      v20 = v42;
    }
    v26 += 80;
    v43 = *(_QWORD *)(a1 + 104) + ((unsigned __int64)*(unsigned int *)(a1 + 112) << 6) - 64;
    *(_QWORD *)(v43 + 32) = v19;
    *(_QWORD *)(v43 + 40) = v20;
    *(_DWORD *)(v43 + 48) = v7;
    *(_DWORD *)(v43 + 52) = v8;
  }
  return a1;
}
