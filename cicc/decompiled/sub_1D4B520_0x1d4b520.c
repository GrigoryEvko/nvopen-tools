// Function: sub_1D4B520
// Address: 0x1d4b520
//
__int64 __fastcall sub_1D4B520(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned int v6; // r12d
  __int64 v7; // r13
  __int64 v8; // rax
  const __m128i *v9; // r15
  __int64 v10; // rax
  __m128i *v11; // rsi
  __m128i *v12; // rax
  __m128i *v13; // rsi
  __m128i *v14; // rax
  __m128i *v15; // rsi
  __int64 v16; // rcx
  unsigned int v17; // ebx
  __int64 v18; // rax
  const __m128i *v19; // r13
  __int64 v20; // r15
  const __m128i *v21; // r8
  __int64 v22; // rax
  _QWORD *v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rcx
  __int16 v27; // dx
  int v28; // edx
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 (*v32)(); // r8
  __int64 v33; // rax
  __m128i *v34; // rsi
  __int32 v35; // edx
  __m128i *v36; // rsi
  __int64 v37; // rax
  __int64 result; // rax
  __m128i *v39; // rsi
  __int64 v40; // [rsp+0h] [rbp-90h]
  __int64 v41; // [rsp+8h] [rbp-88h]
  __int64 v42; // [rsp+10h] [rbp-80h]
  unsigned int v44; // [rsp+28h] [rbp-68h]
  unsigned int v45; // [rsp+28h] [rbp-68h]
  unsigned int v46; // [rsp+2Ch] [rbp-64h]
  __m128i v47; // [rsp+30h] [rbp-60h] BYREF
  const __m128i *v48; // [rsp+40h] [rbp-50h] BYREF
  const __m128i *v49; // [rsp+48h] [rbp-48h]
  __int64 v50; // [rsp+50h] [rbp-40h]

  v7 = a2;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(const __m128i **)a2;
  *(_QWORD *)a2 = 0;
  v42 = v8;
  v10 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  v40 = v10;
  sub_1D4B0A0((const __m128i **)a2, 0, v9);
  v11 = *(__m128i **)(a2 + 8);
  v12 = *(__m128i **)(v7 + 16);
  if ( v11 == v12 )
  {
    sub_1D4B0A0((const __m128i **)v7, v11, v9 + 1);
    v13 = *(__m128i **)(v7 + 8);
    v14 = *(__m128i **)(v7 + 16);
    if ( v13 != v14 )
    {
      if ( !v13 )
      {
LABEL_6:
        v15 = v13 + 1;
        *(_QWORD *)(v7 + 8) = v15;
        if ( v15 != v14 )
        {
LABEL_7:
          *v15 = _mm_loadu_si128(v9 + 3);
          v15 = *(__m128i **)(v7 + 8);
LABEL_8:
          *(_QWORD *)(v7 + 8) = v15 + 1;
          goto LABEL_9;
        }
        goto LABEL_45;
      }
LABEL_5:
      a6 = _mm_loadu_si128(v9 + 2);
      *v13 = a6;
      v13 = *(__m128i **)(v7 + 8);
      v14 = *(__m128i **)(v7 + 16);
      goto LABEL_6;
    }
  }
  else
  {
    if ( v11 )
    {
      a5 = _mm_loadu_si128(v9 + 1);
      *v11 = a5;
      v11 = *(__m128i **)(v7 + 8);
      v12 = *(__m128i **)(v7 + 16);
    }
    v13 = v11 + 1;
    *(_QWORD *)(v7 + 8) = v13;
    if ( v12 != v13 )
      goto LABEL_5;
  }
  sub_1D4B0A0((const __m128i **)v7, v13, v9 + 2);
  v15 = *(__m128i **)(v7 + 8);
  if ( v15 != *(__m128i **)(v7 + 16) )
  {
    if ( !v15 )
      goto LABEL_8;
    goto LABEL_7;
  }
LABEL_45:
  sub_1D4B0A0((const __m128i **)v7, v15, v9 + 3);
LABEL_9:
  v41 = (v42 - (__int64)v9) >> 4;
  v16 = (unsigned int)(v41 - 1);
  if ( *(_BYTE *)(*(_QWORD *)(v9[v16].m128i_i64[0] + 40) + 16LL * v9[v16].m128i_u32[2]) != 111 )
    LODWORD(v16) = (v42 - (__int64)v9) >> 4;
  v46 = v16;
  if ( (_DWORD)v16 != 4 )
  {
    v17 = 4;
    v18 = v7;
    v19 = v9;
    v20 = v18;
    while ( 1 )
    {
      while ( 1 )
      {
        v21 = &v19[v17];
        v22 = *(_QWORD *)(v21->m128i_i64[0] + 88);
        v23 = *(_QWORD **)(v22 + 24);
        if ( *(_DWORD *)(v22 + 32) > 0x40u )
          v23 = (_QWORD *)*v23;
        LODWORD(v24) = (_DWORD)v23;
        if ( ((unsigned __int8)v23 & 7) == 6 )
          break;
        v44 = ((unsigned __int16)v23 >> 3) + 1;
        sub_1D46CE0((const __m128i **)v20, *(__m128i **)(v20 + 8), &v19[v17], &v21[v44]);
        v17 += v44;
LABEL_14:
        if ( v46 == v17 )
          goto LABEL_35;
      }
      if ( (int)v23 < 0 )
      {
        v25 = *(_QWORD *)(v19[4].m128i_i64[0] + 88);
        v26 = *(_QWORD **)(v25 + 24);
        if ( *(_DWORD *)(v25 + 32) > 0x40u )
          v26 = (_QWORD *)*v26;
        v27 = WORD1(v24);
        LODWORD(v24) = (_DWORD)v26;
        v28 = v27 & 0x7FFF;
        if ( v28 )
        {
          LODWORD(v29) = 4;
          do
          {
            v29 = (unsigned int)v29 + ((unsigned __int16)v24 >> 3) + 1;
            v30 = *(_QWORD *)(v19[v29].m128i_i64[0] + 88);
            v24 = *(_QWORD **)(v30 + 24);
            if ( *(_DWORD *)(v30 + 32) > 0x40u )
              v24 = (_QWORD *)*v24;
            --v28;
          }
          while ( v28 );
        }
      }
      v31 = *a1;
      v48 = 0;
      v49 = 0;
      v32 = *(__int64 (**)())(v31 + 216);
      v50 = 0;
      if ( v32 == sub_1D46040
        || (v45 = ((unsigned int)v24 >> 16) & 0x7FFF,
            ((unsigned __int8 (__fastcall *)(__int64 *, const __m128i *, _QWORD, const __m128i **))v32)(
              a1,
              &v19[v17 + 1],
              v45,
              &v48)) )
      {
        sub_16BD130("Could not match memory address.  Inline asm failure!", 1u);
      }
      LOBYTE(v6) = 5;
      v33 = sub_1D38BB0(
              a1[34],
              (v45 << 16) | (8 * (unsigned int)(v49 - v48)) | 6,
              a3,
              v6,
              0,
              1,
              a4,
              *(double *)a5.m128i_i64,
              a6,
              0);
      v34 = *(__m128i **)(v20 + 8);
      v47.m128i_i64[0] = v33;
      v47.m128i_i32[2] = v35;
      if ( v34 == *(__m128i **)(v20 + 16) )
      {
        sub_1D4B3A0((const __m128i **)v20, v34, &v47);
        v36 = *(__m128i **)(v20 + 8);
      }
      else
      {
        if ( v34 )
        {
          a4 = _mm_loadu_si128(&v47);
          *v34 = a4;
          v34 = *(__m128i **)(v20 + 8);
        }
        v36 = v34 + 1;
        *(_QWORD *)(v20 + 8) = v36;
      }
      v17 += 2;
      sub_1D46CE0((const __m128i **)v20, v36, v48, v49);
      if ( !v48 )
        goto LABEL_14;
      j_j___libc_free_0(v48, v50 - (_QWORD)v48);
      if ( v46 == v17 )
      {
LABEL_35:
        v37 = v20;
        v9 = v19;
        v7 = v37;
        break;
      }
    }
  }
  result = v46;
  if ( v46 != v41 )
  {
    v39 = *(__m128i **)(v7 + 8);
    if ( v39 != *(__m128i **)(v7 + 16) )
    {
      if ( v39 )
      {
        result = v42;
        *v39 = _mm_loadu_si128((const __m128i *)(v42 - 16));
        v39 = *(__m128i **)(v7 + 8);
      }
      *(_QWORD *)(v7 + 8) = v39 + 1;
      if ( v9 )
        return j_j___libc_free_0(v9, v40 - (_QWORD)v9);
      return result;
    }
    result = sub_1D4B0A0((const __m128i **)v7, v39, (const __m128i *)(v42 - 16));
  }
  if ( v9 )
    return j_j___libc_free_0(v9, v40 - (_QWORD)v9);
  return result;
}
