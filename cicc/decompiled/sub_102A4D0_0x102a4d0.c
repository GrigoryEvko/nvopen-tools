// Function: sub_102A4D0
// Address: 0x102a4d0
//
__int64 __fastcall sub_102A4D0(unsigned __int8 *a1, __m128i *a2, __int64 *a3)
{
  int v3; // eax
  unsigned __int16 v4; // dx
  __int64 result; // rax
  unsigned __int16 v6; // dx
  __m128i v7; // xmm7
  __m128i v8; // xmm6
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm7
  __m128i v12; // xmm5
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __int64 v20; // rax
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  __m128i v23; // xmm5
  __m128i v24; // xmm3
  __m128i v25; // xmm6
  __m128i v26; // xmm7
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v30; // [rsp+10h] [rbp-70h] BYREF
  __m128i v31; // [rsp+20h] [rbp-60h] BYREF
  __m128i v32; // [rsp+30h] [rbp-50h] BYREF
  __m128i v33; // [rsp+40h] [rbp-40h] BYREF
  __m128i v34; // [rsp+50h] [rbp-30h] BYREF
  __m128i v35[2]; // [rsp+60h] [rbp-20h] BYREF

  v3 = *a1;
  switch ( (_BYTE)v3 )
  {
    case '=':
      v4 = *((_WORD *)a1 + 1);
      if ( ((v4 >> 7) & 6) != 0 || (v4 & 1) != 0 )
      {
        if ( ((v4 >> 7) & 7) != 2 )
        {
LABEL_5:
          a2->m128i_i64[0] = 0;
          a2->m128i_i64[1] = -1;
          a2[1].m128i_i64[0] = 0;
          a2[1].m128i_i64[1] = 0;
          a2[2].m128i_i64[0] = 0;
          a2[2].m128i_i64[1] = 0;
          return 3;
        }
        sub_D665A0(&v30, (__int64)a1);
        v18 = _mm_loadu_si128(&v31);
        v19 = _mm_loadu_si128(&v32);
        *a2 = _mm_loadu_si128(&v30);
        a2[1] = v18;
        a2[2] = v19;
        return 3;
      }
      else
      {
        sub_D665A0(&v30, (__int64)a1);
        v9 = _mm_loadu_si128(&v31);
        v10 = _mm_loadu_si128(&v32);
        *a2 = _mm_loadu_si128(&v30);
        a2[1] = v9;
        a2[2] = v10;
        return 1;
      }
    case '>':
      v6 = *((_WORD *)a1 + 1);
      if ( ((v6 >> 7) & 6) != 0 || (v6 & 1) != 0 )
      {
        if ( ((v6 >> 7) & 7) != 2 )
          goto LABEL_5;
        sub_D66630(&v30, (__int64)a1);
        v11 = _mm_loadu_si128(&v32);
        *a2 = _mm_loadu_si128(&v30);
        v12 = _mm_loadu_si128(&v31);
        a2[2] = v11;
        a2[1] = v12;
        return 3;
      }
      else
      {
        sub_D66630(&v30, (__int64)a1);
        v7 = _mm_loadu_si128(&v31);
        *a2 = _mm_loadu_si128(&v30);
        v8 = _mm_loadu_si128(&v32);
        a2[1] = v7;
        a2[2] = v8;
        return 2;
      }
    case 'Y':
      sub_D666C0(&v30, (__int64)a1);
      v13 = _mm_loadu_si128(&v31);
      v14 = _mm_loadu_si128(&v32);
      *a2 = _mm_loadu_si128(&v30);
      a2[1] = v13;
      a2[2] = v14;
      return 3;
    default:
      v15 = (unsigned int)(v3 - 34);
      if ( (unsigned __int8)v15 > 0x33u )
        goto LABEL_19;
      v16 = 0x8000000000041LL;
      if ( !_bittest64(&v16, v15) )
        goto LABEL_19;
      v17 = sub_D5D560((__int64)a1, a3);
      if ( v17 )
      {
        a2->m128i_i64[0] = v17;
        a2->m128i_i64[1] = 0xBFFFFFFFFFFFFFFELL;
        a2[1].m128i_i64[0] = 0;
        a2[1].m128i_i64[1] = 0;
        a2[2].m128i_i64[0] = 0;
        a2[2].m128i_i64[1] = 0;
        return 2;
      }
      if ( *a1 == 85
        && (v20 = *((_QWORD *)a1 - 4)) != 0
        && !*(_BYTE *)v20
        && *(_QWORD *)(v20 + 24) == *((_QWORD *)a1 + 10)
        && (*(_BYTE *)(v20 + 33) & 0x20) != 0 )
      {
        switch ( *(_DWORD *)(v20 + 36) )
        {
          case 0xCC:
            sub_D669C0(&v33, (__int64)a1, 2u, a3);
            v23 = _mm_loadu_si128(&v34);
            v24 = _mm_loadu_si128(v35);
            *a2 = _mm_loadu_si128(&v33);
            a2[1] = v23;
            a2[2] = v24;
            result = 2;
            break;
          case 0xCD:
          case 0xD2:
          case 0xD3:
            sub_D669C0(&v33, (__int64)a1, 1u, a3);
            v21 = _mm_loadu_si128(v35);
            *a2 = _mm_loadu_si128(&v33);
            v22 = _mm_loadu_si128(&v34);
            a2[2] = v21;
            a2[1] = v22;
            result = 2;
            break;
          case 0xE4:
            sub_D669C0(&v33, (__int64)a1, 0, a3);
            v27 = _mm_loadu_si128(&v34);
            v28 = _mm_loadu_si128(v35);
            *a2 = _mm_loadu_si128(&v33);
            a2[1] = v27;
            a2[2] = v28;
            result = 1;
            break;
          case 0xE6:
            sub_D669C0(&v33, (__int64)a1, 1u, a3);
            v25 = _mm_loadu_si128(&v34);
            v26 = _mm_loadu_si128(v35);
            *a2 = _mm_loadu_si128(&v33);
            a2[1] = v25;
            a2[2] = v26;
            result = 2;
            break;
          default:
            goto LABEL_19;
        }
      }
      else
      {
LABEL_19:
        if ( (unsigned __int8)sub_B46490((__int64)a1) )
          return 3;
        else
          return sub_B46420((__int64)a1);
      }
      break;
  }
  return result;
}
