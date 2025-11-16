// Function: sub_273A0C0
// Address: 0x273a0c0
//
char __fastcall sub_273A0C0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // rax
  __int64 v6; // r11
  __int64 v7; // r10
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  const __m128i *v12; // r10
  const __m128i *v13; // r9
  __int64 v14; // r11
  __int64 v15; // r15
  __int64 v16; // r14
  __m128i *v17; // rdx
  __int64 v18; // rax
  __int32 v19; // ecx
  __int64 v20; // r8
  unsigned __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 v23; // rax
  __m128i v24; // xmm2
  __m128i v25; // xmm1
  __m128i v26; // xmm0
  __m128i v27; // xmm3
  unsigned __int8 v29; // [rsp+Fh] [rbp-91h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  const __m128i *v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+20h] [rbp-80h]
  __int64 v34; // [rsp+20h] [rbp-80h]
  const __m128i *v35; // [rsp+20h] [rbp-80h]
  const __m128i *v36; // [rsp+28h] [rbp-78h]
  const __m128i *v37; // [rsp+28h] [rbp-78h]
  __int64 v38; // [rsp+28h] [rbp-78h]
  __m128i *v39; // [rsp+28h] [rbp-78h]
  __m128i *v40; // [rsp+28h] [rbp-78h]
  __m128i *v41; // [rsp+28h] [rbp-78h]

  LOBYTE(v5) = a4 == 0;
  v29 = a4 == 0 || a5 == 0;
  if ( v29 )
    return (char)v5;
  v6 = (__int64)a1;
  v7 = a2;
  v8 = a4;
  v9 = a5;
  if ( a4 + a5 == 2 )
  {
    v15 = a2;
    v17 = a1;
LABEL_11:
    LODWORD(v5) = v17[3].m128i_i32[0];
    if ( *(_DWORD *)(v15 + 48) == (_DWORD)v5 )
    {
      LODWORD(v5) = *(_DWORD *)(v15 + 56);
      v19 = v17[3].m128i_i32[2];
      if ( (_DWORD)v5 )
      {
        if ( !v19 )
          return (char)v5;
        v20 = *(_QWORD *)v15;
        if ( (_DWORD)v5 == 3 )
        {
          v39 = v17;
          v21 = sub_2739680(*(_QWORD *)v15);
          v17 = v39;
          v20 = v21;
        }
        v22 = v17->m128i_i64[0];
        if ( v19 == 3 )
        {
          v40 = v17;
          v23 = sub_2739680(v17->m128i_i64[0]);
          v17 = v40;
          v22 = v23;
        }
        v41 = v17;
        LOBYTE(v5) = sub_B445A0(v20, v22);
        v17 = v41;
        if ( !(_BYTE)v5 )
          return (char)v5;
      }
      else if ( !v19 )
      {
        if ( **(_BYTE **)(v15 + 8) != 17 )
          v29 = **(_BYTE **)(v15 + 16) != 17;
        v5 = (_BYTE *)v17->m128i_i64[1];
        if ( *v5 == 17 )
          return (char)v5;
        LOBYTE(v5) = *(_BYTE *)v17[1].m128i_i64[0] != 17;
        if ( (unsigned __int8)v5 <= v29 )
          return (char)v5;
      }
    }
    else if ( *(_DWORD *)(v15 + 48) >= (unsigned int)v5 )
    {
      return (char)v5;
    }
    v24 = _mm_loadu_si128(v17);
    v25 = _mm_loadu_si128(v17 + 1);
    v26 = _mm_loadu_si128(v17 + 2);
    *v17 = _mm_loadu_si128((const __m128i *)v15);
    v27 = _mm_loadu_si128(v17 + 3);
    v5 = (_BYTE *)v17[3].m128i_i64[0];
    v17[1] = _mm_loadu_si128((const __m128i *)(v15 + 16));
    v17[2] = _mm_loadu_si128((const __m128i *)(v15 + 32));
    v17[3].m128i_i64[0] = *(_QWORD *)(v15 + 48);
    v17[3].m128i_i32[2] = *(_DWORD *)(v15 + 56);
    *(_QWORD *)(v15 + 48) = v5;
    *(__m128i *)v15 = v24;
    *(_DWORD *)(v15 + 56) = v27.m128i_i32[2];
    *(__m128i *)(v15 + 16) = v25;
    *(__m128i *)(v15 + 32) = v26;
    LOBYTE(v5) = v27.m128i_i8[8];
    return (char)v5;
  }
  if ( a5 >= a4 )
    goto LABEL_9;
LABEL_4:
  v36 = (const __m128i *)v7;
  v31 = v6;
  v10 = v8 / 2;
  v33 = v6 + ((v8 / 2) << 6);
  v11 = sub_2739FD0(v7, a3, v33);
  v12 = v36;
  v13 = (const __m128i *)v33;
  v14 = v31;
  v15 = v11;
  v16 = (v11 - (__int64)v36) >> 6;
  while ( 1 )
  {
    v34 = v14;
    v37 = v13;
    v9 -= v16;
    v32 = sub_27396F0(v13, v12, (const __m128i *)v15);
    LOBYTE(v5) = sub_273A0C0(v34, v37, v32, v10, v16);
    v8 -= v10;
    if ( !v8 )
      break;
    v17 = (__m128i *)v32;
    if ( !v9 )
      break;
    if ( v9 + v8 == 2 )
      goto LABEL_11;
    v7 = v15;
    v6 = (__int64)v32;
    if ( v9 < v8 )
      goto LABEL_4;
LABEL_9:
    v35 = (const __m128i *)v7;
    v38 = v6;
    v16 = v9 / 2;
    v15 = v7 + ((v9 / 2) << 6);
    v18 = sub_2739EE0(v6, v7, v15);
    v14 = v38;
    v12 = v35;
    v13 = (const __m128i *)v18;
    v10 = (v18 - v38) >> 6;
  }
  return (char)v5;
}
