// Function: sub_21E6420
// Address: 0x21e6420
//
unsigned __int64 __fastcall sub_21E6420(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  _DWORD *v6; // rdx
  __m128i *v7; // rdx
  __m128i v8; // xmm0
  _DWORD *v9; // rdx
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  __m128i *v12; // rdx
  __m128i v13; // xmm0
  __m128i *v14; // rdx
  __m128i v15; // xmm0
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __m128i *v20; // rdx
  __m128i v21; // xmm0
  __m128i *v22; // rdx
  __m128i v23; // xmm0
  __m128i *v24; // rdx
  __m128i v25; // xmm0
  __m128i *v26; // rdx
  __m128i v27; // xmm0
  __m128i *v28; // rdx
  __m128i v29; // xmm0
  __m128i *v30; // rdx
  __m128i v31; // xmm0
  __m128i *v32; // rdx
  __m128i v33; // xmm0
  size_t v34; // rdx
  char *v35; // rsi

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  result = (unsigned __int8)v4 >> 4;
  if ( (((unsigned int)v4 >> 4) & 0xF) == 1 )
  {
    v9 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v9;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".cta", 4u);
    }
    else
    {
      *v9 = 1635017518;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  else if ( (_BYTE)result == 2 )
  {
    v6 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v6;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".sys", 4u);
    }
    else
    {
      *v6 = 1937339182;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  switch ( BYTE2(v4) )
  {
    case 0:
      v10 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v10 <= 0x15u )
      {
        v34 = 22;
        v35 = ".exch.L2::cache_hint.b";
        goto LABEL_37;
      }
      si128 = _mm_load_si128((const __m128i *)&xmmword_435F5A0);
      v10[1].m128i_i32[0] = 1953393000;
      v10[1].m128i_i16[2] = 25134;
      *v10 = si128;
      *(_QWORD *)(a3 + 24) += 22LL;
      result = 25134;
      break;
    case 1:
      v12 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v12;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".add.L2::cache_hint.u";
        goto LABEL_37;
      }
      v13 = _mm_load_si128((const __m128i *)&xmmword_435F590);
      v12[1].m128i_i32[0] = 779382377;
      v12[1].m128i_i8[4] = 117;
      *v12 = v13;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 3:
      v14 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v14;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".and.L2::cache_hint.b";
        goto LABEL_37;
      }
      v15 = _mm_load_si128((const __m128i *)&xmmword_435F5C0);
      v14[1].m128i_i32[0] = 779382377;
      v14[1].m128i_i8[4] = 98;
      *v14 = v15;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 5:
      v16 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v16;
      if ( result <= 0x13 )
      {
        v34 = 20;
        v35 = ".or.L2::cache_hint.b";
        goto LABEL_37;
      }
      v17 = _mm_load_si128((const __m128i *)&xmmword_435F5B0);
      v16[1].m128i_i32[0] = 1647211630;
      *v16 = v17;
      *(_QWORD *)(a3 + 24) += 20LL;
      break;
    case 6:
      v18 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v18;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".xor.L2::cache_hint.b";
        goto LABEL_37;
      }
      v19 = _mm_load_si128((const __m128i *)&xmmword_435F5D0);
      v18[1].m128i_i32[0] = 779382377;
      v18[1].m128i_i8[4] = 98;
      *v18 = v19;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 7:
      v20 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v20;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".max.L2::cache_hint.s";
        goto LABEL_37;
      }
      v21 = _mm_load_si128((const __m128i *)&xmmword_435F5E0);
      v20[1].m128i_i32[0] = 779382377;
      v20[1].m128i_i8[4] = 115;
      *v20 = v21;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 8:
      v26 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v26;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".min.L2::cache_hint.s";
        goto LABEL_37;
      }
      v27 = _mm_load_si128((const __m128i *)&xmmword_435F5F0);
      v26[1].m128i_i32[0] = 779382377;
      v26[1].m128i_i8[4] = 115;
      *v26 = v27;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 9:
      v28 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v28;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".max.L2::cache_hint.u";
        goto LABEL_37;
      }
      v29 = _mm_load_si128((const __m128i *)&xmmword_435F5E0);
      v28[1].m128i_i32[0] = 779382377;
      v28[1].m128i_i8[4] = 117;
      *v28 = v29;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 0xA:
      v30 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v30;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".min.L2::cache_hint.u";
        goto LABEL_37;
      }
      v31 = _mm_load_si128((const __m128i *)&xmmword_435F5F0);
      v30[1].m128i_i32[0] = 779382377;
      v30[1].m128i_i8[4] = 117;
      *v30 = v31;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 0xB:
      v32 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v32;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".add.L2::cache_hint.f";
        goto LABEL_37;
      }
      v33 = _mm_load_si128((const __m128i *)&xmmword_435F590);
      v32[1].m128i_i32[0] = 779382377;
      v32[1].m128i_i8[4] = 102;
      *v32 = v33;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 0xC:
      v22 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v22;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".inc.L2::cache_hint.u";
        goto LABEL_37;
      }
      v23 = _mm_load_si128((const __m128i *)&xmmword_435F600);
      v22[1].m128i_i32[0] = 779382377;
      v22[1].m128i_i8[4] = 117;
      *v22 = v23;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 0xD:
      v24 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v24;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".dec.L2::cache_hint.u";
        goto LABEL_37;
      }
      v25 = _mm_load_si128((const __m128i *)&xmmword_435F610);
      v24[1].m128i_i32[0] = 779382377;
      v24[1].m128i_i8[4] = 117;
      *v24 = v25;
      *(_QWORD *)(a3 + 24) += 21LL;
      break;
    case 0xE:
      v7 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v7;
      if ( result <= 0x14 )
      {
        v34 = 21;
        v35 = ".cas.L2::cache_hint.b";
LABEL_37:
        result = sub_16E7EE0(a3, v35, v34);
      }
      else
      {
        v8 = _mm_load_si128((const __m128i *)&xmmword_435F620);
        v7[1].m128i_i32[0] = 779382377;
        v7[1].m128i_i8[4] = 98;
        *v7 = v8;
        *(_QWORD *)(a3 + 24) += 21LL;
      }
      break;
    default:
      return result;
  }
  return result;
}
