// Function: sub_39D5BB0
// Address: 0x39d5bb0
//
void __fastcall sub_39D5BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int32 v7; // ebx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // edi
  __m128i *v14; // r9
  __int64 v15; // r15
  unsigned int *v16; // rbx
  __m128i *v17; // r8
  unsigned int v18; // edi
  unsigned int v19; // r8d
  _WORD *v20; // rax
  _WORD *v21; // rbx
  __m128i *v22; // rdi
  __m128i *v23; // rsi
  const __m128i *v24; // r13
  __int64 v25; // rdi
  const __m128i *v26; // r14
  unsigned __int64 *v27; // r13
  signed __int64 v28; // r15
  __int64 v29; // rax
  __m128i *v30; // rbx
  unsigned __int64 *v31; // r13
  __int64 v32; // [rsp+10h] [rbp-120h]
  unsigned int *v33; // [rsp+10h] [rbp-120h]
  int v34; // [rsp+18h] [rbp-118h]
  __m128i *v35; // [rsp+18h] [rbp-118h]
  unsigned int v36; // [rsp+18h] [rbp-118h]
  __int64 v37[2]; // [rsp+30h] [rbp-100h] BYREF
  void (__fastcall *v38)(__int64 *, __int64 *, __int64, __int64, __int64, unsigned __int64 **); // [rsp+40h] [rbp-F0h]
  void (__fastcall *v39)(__int64 *, unsigned __int64 **); // [rsp+48h] [rbp-E8h]
  unsigned __int64 *v40; // [rsp+50h] [rbp-E0h] BYREF
  __m128i *v41; // [rsp+58h] [rbp-D8h]
  const __m128i *v42; // [rsp+60h] [rbp-D0h]
  __int64 v43; // [rsp+68h] [rbp-C8h]
  int v44; // [rsp+70h] [rbp-C0h]
  __int8 **v45; // [rsp+78h] [rbp-B8h]
  __m128i v46; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+90h] [rbp-A0h] BYREF
  __int8 *v48; // [rsp+98h] [rbp-98h] BYREF
  __m128i v49; // [rsp+A0h] [rbp-90h] BYREF
  __int8 *v50; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v51; // [rsp+B8h] [rbp-78h] BYREF
  __int8 *v52; // [rsp+C8h] [rbp-68h] BYREF
  __m128i v53; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v54[4]; // [rsp+E8h] [rbp-48h] BYREF

  v7 = 0;
  *(_BYTE *)(a2 + 25) = (**(_QWORD **)(*(_QWORD *)a3 + 352LL) & 4LL) != 0;
  v34 = *(_DWORD *)(a3 + 32);
  if ( v34 )
  {
    do
    {
      while ( 1 )
      {
        v46.m128i_i64[1] = 0;
        v49.m128i_i8[8] = 0;
        v15 = v7 & 0x7FFFFFFF;
        v48 = &v49.m128i_i8[8];
        v47 = 0;
        v49.m128i_i64[0] = 0;
        v51 = 0u;
        v52 = &v53.m128i_i8[8];
        v53.m128i_i64[0] = 0;
        v53.m128i_i8[8] = 0;
        v54[0] = 0u;
        v46.m128i_i32[0] = v7;
        if ( *(_DWORD *)(a3 + 72) <= (unsigned int)v15
          || !*(_QWORD *)(*(_QWORD *)(a3 + 64) + 32LL * (unsigned int)v15 + 8) )
        {
          break;
        }
        if ( ++v7 == v34 )
          goto LABEL_22;
      }
      v8 = v7 | 0x80000000;
      v44 = 1;
      v40 = (unsigned __int64 *)&unk_49EFBE0;
      v43 = 0;
      v42 = 0;
      v41 = 0;
      v45 = &v48;
      sub_1F4AA90(v37, v8, a3, a4);
      if ( !v38 )
        sub_4263D6(v37, v8, v9);
      v39(v37, &v40);
      if ( v38 )
        v38(v37, v37, 3, v10, v11, &v40);
      sub_16E7BC0((__int64 *)&v40);
      v12 = *(_QWORD *)(a3 + 208) + 40 * v15;
      if ( *(_DWORD *)(v12 + 16) )
      {
        if ( !*(_DWORD *)v12 )
        {
          v13 = **(_DWORD **)(v12 + 8);
          if ( v13 )
            sub_39CF6E0(v13, (__int64)&v52, a4);
        }
      }
      v14 = *(__m128i **)(a2 + 40);
      if ( v14 == *(__m128i **)(a2 + 48) )
      {
        sub_39D5130((unsigned __int64 *)(a2 + 32), *(_QWORD *)(a2 + 40), &v46);
      }
      else
      {
        if ( v14 )
        {
          v32 = *(_QWORD *)(a2 + 40);
          *v14 = _mm_loadu_si128(&v46);
          v14[1].m128i_i64[0] = v47;
          v14[1].m128i_i64[1] = (__int64)&v14[2].m128i_i64[1];
          sub_39CF630(&v14[1].m128i_i64[1], v48, (__int64)&v48[v49.m128i_i64[0]]);
          *(__m128i *)(v32 + 56) = _mm_loadu_si128(&v51);
          *(_QWORD *)(v32 + 72) = v32 + 88;
          sub_39CF630((__int64 *)(v32 + 72), v52, (__int64)&v52[v53.m128i_i64[0]]);
          *(__m128i *)(v32 + 104) = _mm_loadu_si128(v54);
          v14 = *(__m128i **)(a2 + 40);
        }
        *(_QWORD *)(a2 + 40) = (char *)v14 + 120;
      }
      if ( v52 != (__int8 *)&v53.m128i_u64[1] )
        j_j___libc_free_0((unsigned __int64)v52);
      if ( v48 != (__int8 *)&v49.m128i_u64[1] )
        j_j___libc_free_0((unsigned __int64)v48);
      ++v7;
    }
    while ( v7 != v34 );
  }
LABEL_22:
  v16 = *(unsigned int **)(a3 + 360);
  v33 = *(unsigned int **)(a3 + 368);
  while ( v33 != v16 )
  {
    v18 = *v16;
    v19 = v16[1];
    LOBYTE(v47) = 0;
    v46 = (__m128i)(unsigned __int64)&v47;
    v36 = v19;
    v49 = 0u;
    v50 = &v51.m128i_i8[8];
    v51.m128i_i64[0] = 0;
    v51.m128i_i8[8] = 0;
    v53 = 0u;
    sub_39CF6E0(v18, (__int64)&v46, a4);
    if ( v36 )
    {
      sub_39CF6E0(v36, (__int64)&v50, a4);
      v17 = *(__m128i **)(a2 + 64);
      if ( v17 != *(__m128i **)(a2 + 72) )
      {
LABEL_25:
        if ( v17 )
        {
          v35 = v17;
          v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
          sub_39CF630(v17->m128i_i64, v46.m128i_i64[0], v46.m128i_i64[0] + v46.m128i_i64[1]);
          v35[2] = _mm_loadu_si128(&v49);
          v35[3].m128i_i64[0] = (__int64)v35[4].m128i_i64;
          sub_39CF630(v35[3].m128i_i64, v50, (__int64)&v50[v51.m128i_i64[0]]);
          v35[5] = _mm_loadu_si128(&v53);
          v17 = *(__m128i **)(a2 + 64);
        }
        *(_QWORD *)(a2 + 64) = v17 + 6;
        goto LABEL_28;
      }
    }
    else
    {
      v17 = *(__m128i **)(a2 + 64);
      if ( v17 != *(__m128i **)(a2 + 72) )
        goto LABEL_25;
    }
    sub_39D5540((unsigned __int64 *)(a2 + 56), v17, &v46);
LABEL_28:
    if ( v50 != (__int8 *)&v51.m128i_u64[1] )
      j_j___libc_free_0((unsigned __int64)v50);
    if ( (__int64 *)v46.m128i_i64[0] != &v47 )
      j_j___libc_free_0(v46.m128i_u64[0]);
    v16 += 2;
  }
  if ( *(_BYTE *)(a3 + 152) )
  {
    v20 = (_WORD *)sub_1E6A620((_QWORD *)a3);
    v40 = 0;
    v41 = 0;
    v21 = v20;
    v42 = 0;
    if ( *v20 )
    {
      do
      {
        v23 = &v46;
        LOBYTE(v47) = 0;
        v46 = (__m128i)(unsigned __int64)&v47;
        v49 = 0u;
        sub_39CF6E0((unsigned __int16)*v21, (__int64)&v46, a4);
        v24 = v41;
        if ( v41 == v42 )
        {
          v23 = v41;
          sub_39D5900((unsigned __int64 *)&v40, v41, &v46);
        }
        else
        {
          if ( v41 )
          {
            v22 = v41;
            v41->m128i_i64[0] = (__int64)v41[1].m128i_i64;
            v23 = (__m128i *)v46.m128i_i64[0];
            sub_39CF630(v22->m128i_i64, v46.m128i_i64[0], v46.m128i_i64[0] + v46.m128i_i64[1]);
            v24[2] = _mm_loadu_si128(&v49);
            v24 = v41;
          }
          v41 = (__m128i *)&v24[3];
        }
        v25 = v46.m128i_i64[0];
        if ( (__int64 *)v46.m128i_i64[0] != &v47 )
        {
          v23 = (__m128i *)(v47 + 1);
          j_j___libc_free_0(v46.m128i_u64[0]);
        }
        ++v21;
      }
      while ( *v21 );
      if ( !*(_BYTE *)(a2 + 104) )
      {
        v26 = v41;
        v27 = v40;
        *(_QWORD *)(a2 + 80) = 0;
        *(_QWORD *)(a2 + 88) = 0;
        *(_QWORD *)(a2 + 96) = 0;
        v28 = (char *)v26 - (char *)v27;
        if ( v26 == (const __m128i *)v27 )
        {
          v30 = 0;
        }
        else
        {
          if ( (unsigned __int64)((char *)v26 - (char *)v27) > 0x7FFFFFFFFFFFFFE0LL )
            sub_4261EA(v25, v23, 0x7FFFFFFFFFFFFFE0LL);
          v29 = sub_22077B0((char *)v26 - (char *)v27);
          v26 = v41;
          v27 = v40;
          v30 = (__m128i *)v29;
        }
        goto LABEL_53;
      }
    }
    else if ( !*(_BYTE *)(a2 + 104) )
    {
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v30 = 0;
LABEL_53:
      *(_QWORD *)(a2 + 80) = v30;
      *(_QWORD *)(a2 + 88) = v30;
      *(_QWORD *)(a2 + 96) = (char *)v30 + v28;
      if ( v27 == (unsigned __int64 *)v26 )
      {
        v31 = (unsigned __int64 *)v26;
      }
      else
      {
        do
        {
          if ( v30 )
          {
            v30->m128i_i64[0] = (__int64)v30[1].m128i_i64;
            sub_39CF630(v30->m128i_i64, (_BYTE *)*v27, *v27 + v27[1]);
            v30[2] = _mm_loadu_si128((const __m128i *)v27 + 2);
          }
          v27 += 6;
          v30 += 3;
        }
        while ( v26 != (const __m128i *)v27 );
        v26 = v41;
        v31 = v40;
      }
      *(_QWORD *)(a2 + 88) = v30;
      *(_BYTE *)(a2 + 104) = 1;
      goto LABEL_60;
    }
    sub_39CF9D0(a2 + 80, &v40);
    v26 = v41;
    v31 = v40;
LABEL_60:
    if ( v26 != (const __m128i *)v31 )
    {
      do
      {
        if ( (unsigned __int64 *)*v31 != v31 + 2 )
          j_j___libc_free_0(*v31);
        v31 += 6;
      }
      while ( v26 != (const __m128i *)v31 );
      v31 = v40;
    }
    if ( v31 )
      j_j___libc_free_0((unsigned __int64)v31);
  }
}
