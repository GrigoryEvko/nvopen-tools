// Function: sub_14113F0
// Address: 0x14113f0
//
char __fastcall sub_14113F0(__int64 a1, __m128i *a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  char v6; // al
  __int64 v7; // rdx
  unsigned int v8; // eax
  char result; // al
  __int64 v10; // rdx
  unsigned int v11; // eax
  __m128i v12; // xmm4
  __m128i v13; // xmm5
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  char v22; // r8
  __int64 v23; // rax
  unsigned int v24; // eax
  _QWORD *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __m128i v29; // [rsp+10h] [rbp-40h] BYREF
  __m128i v30; // [rsp+20h] [rbp-30h] BYREF
  __int64 v31; // [rsp+30h] [rbp-20h]

  v6 = *(_BYTE *)(a1 + 16);
  switch ( v6 )
  {
    case '6':
      v7 = *(unsigned __int16 *)(a1 + 18);
      v8 = (unsigned int)v7 >> 7;
      if ( (((unsigned int)v7 >> 7) & 6) == 0 )
      {
        v7 &= 1u;
        if ( !(_DWORD)v7 )
        {
          sub_141EB40(&v29, a1, v7, a4, a5);
          v14 = _mm_loadu_si128(&v29);
          v15 = _mm_loadu_si128(&v30);
          a2[2].m128i_i64[0] = v31;
          *a2 = v14;
          a2[1] = v15;
          return 5;
        }
      }
      if ( (v8 & 7) != 2 )
        goto LABEL_5;
      sub_141EB40(&v29, a1, v7, a4, a5);
      goto LABEL_20;
    case '7':
      v10 = *(unsigned __int16 *)(a1 + 18);
      v11 = (unsigned int)v10 >> 7;
      if ( (((unsigned int)v10 >> 7) & 6) != 0 || (v10 &= 1u, (_DWORD)v10) )
      {
        if ( (v11 & 7) != 2 )
        {
LABEL_5:
          a2->m128i_i64[0] = 0;
          result = 7;
          a2->m128i_i64[1] = -1;
          a2[1].m128i_i64[0] = 0;
          a2[1].m128i_i64[1] = 0;
          a2[2].m128i_i64[0] = 0;
          return result;
        }
        sub_141EDF0(&v29, a1, v10, a4, a5);
LABEL_20:
        v20 = _mm_loadu_si128(&v29);
        v21 = _mm_loadu_si128(&v30);
        a2[2].m128i_i64[0] = v31;
        *a2 = v20;
        a2[1] = v21;
        return 7;
      }
      sub_141EDF0(&v29, a1, v10, a4, a5);
LABEL_11:
      v12 = _mm_loadu_si128(&v29);
      v13 = _mm_loadu_si128(&v30);
      a2[2].m128i_i64[0] = v31;
      *a2 = v12;
      a2[1] = v13;
      return 6;
    case 'R':
      sub_141F0A0(&v29);
      v16 = _mm_loadu_si128(&v29);
      v17 = _mm_loadu_si128(&v30);
      a2[2].m128i_i64[0] = v31;
      *a2 = v16;
      a2[1] = v17;
      return 7;
  }
  v18 = sub_140B650(a1, a3);
  if ( v18 )
  {
    v19 = *(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
    a2->m128i_i64[1] = -1;
    a2[1].m128i_i64[0] = 0;
    a2->m128i_i64[0] = v19;
    a2[1].m128i_i64[1] = 0;
    a2[2].m128i_i64[0] = 0;
    return 6;
  }
  if ( *(_BYTE *)(a1 + 16) != 78 )
    goto LABEL_22;
  v23 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v23 + 16) || (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
    goto LABEL_22;
  v24 = *(_DWORD *)(v23 + 36);
  if ( v24 == 114 )
  {
LABEL_32:
    v25 = a3;
    v26 = 1;
    v27 = a1 | 4;
    goto LABEL_30;
  }
  if ( v24 > 0x72 )
  {
    if ( v24 - 116 > 1 )
      goto LABEL_22;
    goto LABEL_32;
  }
  if ( v24 == 113 )
  {
    v25 = a3;
    v26 = 2;
    v27 = a1 | 4;
LABEL_30:
    sub_141F820(&v29, v27, v26, v25);
    goto LABEL_11;
  }
LABEL_22:
  v22 = sub_15F3040(a1);
  result = 7;
  if ( !v22 )
    return 4 - (((unsigned __int8)sub_15F2ED0(a1) == 0) - 1);
  return result;
}
