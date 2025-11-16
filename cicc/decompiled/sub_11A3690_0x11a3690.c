// Function: sub_11A3690
// Address: 0x11a3690
//
unsigned __int8 *__fastcall sub_11A3690(
        const __m128i *a1,
        unsigned __int8 *a2,
        unsigned int a3,
        __int64 a4,
        int a5,
        __int64 a6)
{
  __int64 **v6; // r15
  unsigned __int8 *result; // rax
  int *v11; // rax
  int v12; // eax
  int v13; // eax
  __int64 v14; // rdx
  unsigned int v15; // r9d
  __m128i v16; // xmm6
  __int64 v17; // rax
  unsigned __int64 v18; // xmm7_8
  __m128i v19; // xmm0
  int v20; // esi
  __m128i v21; // xmm2
  unsigned __int64 v22; // xmm3_8
  __int64 v23; // rax
  __m128i v24; // xmm4
  int v25; // eax
  __m128i v26; // xmm6
  __int64 v27; // rax
  unsigned __int64 v28; // xmm7_8
  __m128i v29; // xmm0
  unsigned int v30; // eax
  int v31; // eax
  bool v32; // zf
  unsigned __int8 *v33; // rax
  unsigned int v34; // eax
  int v35; // eax
  int v36; // esi
  unsigned int v37; // eax
  __int64 v38; // rax
  __m128i *v39; // rdi
  const __m128i *v40; // rsi
  __int64 v41; // rcx
  __int64 *v42; // r8
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // rdx
  unsigned __int8 *v46; // rax
  unsigned int v49; // [rsp+18h] [rbp-B8h]
  __int64 v50; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v51; // [rsp+28h] [rbp-A8h]
  __int64 v52; // [rsp+30h] [rbp-A0h]
  __int64 v53; // [rsp+38h] [rbp-98h]
  __int64 v54; // [rsp+48h] [rbp-88h] BYREF
  __m128i v55; // [rsp+50h] [rbp-80h] BYREF
  __m128i v56; // [rsp+60h] [rbp-70h]
  unsigned __int64 v57; // [rsp+70h] [rbp-60h]
  __int64 v58; // [rsp+78h] [rbp-58h]
  __m128i v59; // [rsp+80h] [rbp-50h]
  __int64 v60; // [rsp+90h] [rbp-40h]

  v6 = (__int64 **)*((_QWORD *)a2 + 1);
  if ( !a3 )
  {
    if ( (unsigned int)*a2 - 12 > 1 )
      return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)a2 + 1));
    return 0;
  }
  v11 = (int *)sub_C94E20((__int64)qword_4F862D0);
  if ( v11 )
    v12 = *v11;
  else
    v12 = qword_4F862D0[2];
  if ( a5 == v12 )
    return 0;
  v13 = *a2;
  if ( (unsigned __int8)v13 > 0x1Cu )
  {
    v14 = *((_QWORD *)a2 + 2);
    if ( v14 && !*(_QWORD *)(v14 + 8) )
    {
      v15 = a5 + 1;
      if ( v13 != 85 )
      {
        if ( v13 == 86 )
        {
          v55.m128i_i64[0] = 1023;
          v54 = 1023;
          if ( !(unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 2u, a3, (__int64)&v55, v15)
            && !(unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 1u, a3, (__int64)&v54, a5 + 1) )
          {
            if ( ((unsigned int)v54 & a3) == 0 )
              return *(unsigned __int8 **)(sub_986520((__int64)a2) + 64);
            if ( (v55.m128i_i32[0] & a3) == 0 )
              return *(unsigned __int8 **)(sub_986520((__int64)a2) + 32);
            v45 = BYTE5(v54);
            v20 = v55.m128i_i32[0] | v54;
            if ( BYTE5(v54) == v55.m128i_i8[5] )
            {
              if ( BYTE5(v54) && BYTE4(v54) != v55.m128i_i8[4] )
                v45 = 0;
            }
            else
            {
              v45 = 0;
            }
            v51 = (v45 << 40) | v55.m128i_i32[0] | (unsigned int)v54 | v54 & 0xFFFF00FF00000000LL;
            *(_DWORD *)a4 = v20;
            *(_WORD *)(a4 + 4) = WORD2(v51);
            return sub_11A05B0(v6, a3 & v20);
          }
        }
        else
        {
          if ( v13 != 41 )
          {
            v16 = _mm_loadu_si128(a1 + 7);
            v17 = a1[10].m128i_i64[0];
            v18 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
            v55 = _mm_loadu_si128(a1 + 6);
            v19 = _mm_loadu_si128(a1 + 9);
            v60 = v17;
            v57 = v18;
            v56 = v16;
            v58 = a6;
            v59 = v19;
            v50 = sub_9B4030((__int64 *)a2, ~(_WORD)a3 & 0x3FF, v15, &v55);
            v20 = v50;
            *(_DWORD *)a4 = v50;
            *(_WORD *)(a4 + 4) = WORD2(v50);
            return sub_11A05B0(v6, a3 & v20);
          }
          v30 = sub_C65050(a3);
          if ( !(unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 0, v30, a4, a5 + 1) )
          {
            v31 = sub_C65050(*(_DWORD *)a4);
            v32 = *(_BYTE *)(a4 + 5) == 0;
            *(_DWORD *)a4 = v31;
            v20 = v31;
            if ( !v32 )
            {
              *(_BYTE *)(a4 + 4) ^= 1u;
              *(_BYTE *)(a4 + 5) = 1;
            }
            return sub_11A05B0(v6, a3 & v20);
          }
        }
        return a2;
      }
      v49 = a5 + 1;
      v25 = sub_B49240((__int64)a2);
      if ( v25 != 26 )
      {
        if ( v25 == 170 )
        {
          v34 = sub_C650C0(a3);
          if ( !(unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 0, v34, a4, v49) )
          {
            v35 = *(_DWORD *)a4;
            if ( (*(_DWORD *)a4 & 0x20) != 0 )
            {
              v35 |= 0x40u;
              *(_DWORD *)a4 = v35;
            }
            if ( (v35 & 4) != 0 )
            {
              BYTE1(v35) |= 2u;
              *(_DWORD *)a4 = v35;
            }
            if ( (v35 & 0x10) != 0 )
            {
              LOBYTE(v35) = v35 | 0x80;
              *(_DWORD *)a4 = v35;
            }
            if ( (v35 & 8) != 0 )
            {
              BYTE1(v35) |= 1u;
              *(_DWORD *)a4 = v35;
            }
            v36 = *(_DWORD *)a4;
            *(_WORD *)(a4 + 4) = 256;
            v20 = v36 & 0x3C3;
            *(_DWORD *)a4 = v20;
            return sub_11A05B0(v6, a3 & v20);
          }
          return a2;
        }
        if ( v25 != 8 )
        {
          v26 = _mm_loadu_si128(a1 + 7);
          v27 = a1[10].m128i_i64[0];
          v28 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v55 = _mm_loadu_si128(a1 + 6);
          v29 = _mm_loadu_si128(a1 + 9);
          v60 = v27;
          v57 = v28;
          v56 = v26;
          v58 = a6;
          v59 = v29;
          v52 = sub_9B4030((__int64 *)a2, ~(_WORD)a3 & 0x3FF, v49, &v55);
          v20 = v52;
          *(_DWORD *)a4 = v52;
          *(_WORD *)(a4 + 4) = WORD2(v52);
          return sub_11A05B0(v6, a3 & v20);
        }
        if ( (unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 0, a3, a4, v49) )
          return a2;
LABEL_34:
        v20 = *(_DWORD *)a4;
        return sub_11A05B0(v6, a3 & v20);
      }
      v37 = sub_C65100(a3);
      if ( (unsigned __int8)sub_11A3550((__int64)a1, (__int64)a2, 0, v37, a4, v49) )
        return a2;
      if ( (a3 & 0x3C0) == 0 )
      {
        v46 = sub_AD8DD0((__int64)v6, -1.0);
        sub_11A11E0((__int64)a2, 1u, (__int64)v46);
        return a2;
      }
      if ( (a3 & 0x3C) == 0 )
      {
        v33 = sub_AD9290((__int64)v6, 0);
        sub_11A11E0((__int64)a2, 1u, (__int64)v33);
        return a2;
      }
      v38 = sub_986520((__int64)a2);
      v39 = &v55;
      v40 = a1 + 6;
      v41 = 18;
      v42 = *(__int64 **)(v38 + 32);
      while ( v41 )
      {
        v39->m128i_i32[0] = v40->m128i_i32[0];
        v40 = (const __m128i *)((char *)v40 + 4);
        v39 = (__m128i *)((char *)v39 + 4);
        --v41;
      }
      v58 = a6;
      v43 = sub_9B4030(v42, 1023, v49, &v55);
      v44 = *(_DWORD *)a4;
      if ( (*(_DWORD *)a4 & 0x60) != 0 )
      {
        v44 |= 0x60u;
        *(_DWORD *)a4 = v44;
      }
      if ( (v44 & 0x90) != 0 )
      {
        LOBYTE(v44) = v44 | 0x90;
        *(_DWORD *)a4 = v44;
      }
      if ( (v44 & 0x108) != 0 )
      {
        v44 |= 0x108u;
        *(_DWORD *)a4 = v44;
      }
      if ( (v44 & 0x204) != 0 )
        *(_DWORD *)a4 = v44 | 0x204;
      *(_WORD *)(a4 + 4) = (unsigned __int64)(v43 << 16) >> 48;
      if ( (v43 & 0x3C3) != 0 )
      {
        if ( !*(_BYTE *)(a4 + 5) )
        {
          if ( (v43 & 0x3F) != 0 )
            goto LABEL_34;
          goto LABEL_64;
        }
        if ( !*(_BYTE *)(a4 + 4) || (*(_DWORD *)a4 &= 0x3Fu, (v43 & 0x3F) == 0) )
        {
LABEL_64:
          *(_DWORD *)a4 &= 0x3C3u;
          goto LABEL_34;
        }
      }
      else
      {
        *(_DWORD *)a4 &= 0x3Fu;
        if ( (v43 & 0x3F) == 0 )
          goto LABEL_64;
        if ( !*(_BYTE *)(a4 + 5) )
          goto LABEL_34;
      }
      if ( *(_BYTE *)(a4 + 4) )
        goto LABEL_34;
      goto LABEL_64;
    }
    return 0;
  }
  v21 = _mm_loadu_si128(a1 + 7);
  v22 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v23 = a1[10].m128i_i64[0];
  v24 = _mm_loadu_si128(a1 + 9);
  v55 = _mm_loadu_si128(a1 + 6);
  v57 = v22;
  v60 = v23;
  v58 = a6;
  v56 = v21;
  v59 = v24;
  v53 = sub_9B4030((__int64 *)a2, 1023, a5 + 1, &v55);
  *(_DWORD *)a4 = v53;
  *(_WORD *)(a4 + 4) = WORD2(v53);
  result = sub_11A05B0(v6, a3 & (unsigned int)v53);
  if ( result == a2 )
    return 0;
  return result;
}
