// Function: sub_35F0A40
// Address: 0x35f0a40
//
unsigned __int64 __fastcall sub_35F0A40(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rbx
  unsigned __int64 result; // rax
  _DWORD *v7; // rdx
  __m128i *v8; // rdx
  __m128i v9; // xmm0
  _DWORD *v10; // rdx
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  __m128i *v15; // rdx
  __m128i v16; // xmm0
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __m128i *v19; // rdx
  __m128i v20; // xmm0
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  __m128i *v27; // rdx
  __m128i v28; // xmm0
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  __m128i *v31; // rdx
  __m128i v32; // xmm0
  __m128i *v33; // rdx
  __m128i v34; // xmm0
  size_t v35; // rdx
  char *v36; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  result = (unsigned __int8)v5 >> 4;
  if ( (((unsigned int)v5 >> 4) & 0xF) == 1 )
  {
    v10 = *(_DWORD **)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - (_QWORD)v10;
    if ( result <= 3 )
    {
      result = sub_CB6200(a4, (unsigned __int8 *)".cta", 4u);
    }
    else
    {
      *v10 = 1635017518;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  else if ( (_BYTE)result == 2 )
  {
    v7 = *(_DWORD **)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - (_QWORD)v7;
    if ( result <= 3 )
    {
      result = sub_CB6200(a4, (unsigned __int8 *)".sys", 4u);
    }
    else
    {
      *v7 = 1937339182;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  switch ( BYTE2(v5) )
  {
    case 0:
      v11 = *(__m128i **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 0x15u )
      {
        v35 = 22;
        v36 = ".exch.L2::cache_hint.b";
        goto LABEL_37;
      }
      si128 = _mm_load_si128((const __m128i *)&xmmword_435F5A0);
      v11[1].m128i_i32[0] = 1953393000;
      v11[1].m128i_i16[2] = 25134;
      *v11 = si128;
      *(_QWORD *)(a4 + 32) += 22LL;
      result = 25134;
      break;
    case 1:
      v13 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v13;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".add.L2::cache_hint.u";
        goto LABEL_37;
      }
      v14 = _mm_load_si128((const __m128i *)&xmmword_435F590);
      v13[1].m128i_i32[0] = 779382377;
      v13[1].m128i_i8[4] = 117;
      *v13 = v14;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 3:
      v15 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v15;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".and.L2::cache_hint.b";
        goto LABEL_37;
      }
      v16 = _mm_load_si128((const __m128i *)&xmmword_435F5C0);
      v15[1].m128i_i32[0] = 779382377;
      v15[1].m128i_i8[4] = 98;
      *v15 = v16;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 5:
      v17 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v17;
      if ( result <= 0x13 )
      {
        v35 = 20;
        v36 = ".or.L2::cache_hint.b";
        goto LABEL_37;
      }
      v18 = _mm_load_si128((const __m128i *)&xmmword_435F5B0);
      v17[1].m128i_i32[0] = 1647211630;
      *v17 = v18;
      *(_QWORD *)(a4 + 32) += 20LL;
      break;
    case 6:
      v19 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v19;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".xor.L2::cache_hint.b";
        goto LABEL_37;
      }
      v20 = _mm_load_si128((const __m128i *)&xmmword_435F5D0);
      v19[1].m128i_i32[0] = 779382377;
      v19[1].m128i_i8[4] = 98;
      *v19 = v20;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 7:
      v21 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v21;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".max.L2::cache_hint.s";
        goto LABEL_37;
      }
      v22 = _mm_load_si128((const __m128i *)&xmmword_435F5E0);
      v21[1].m128i_i32[0] = 779382377;
      v21[1].m128i_i8[4] = 115;
      *v21 = v22;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 8:
      v27 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v27;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".min.L2::cache_hint.s";
        goto LABEL_37;
      }
      v28 = _mm_load_si128((const __m128i *)&xmmword_435F5F0);
      v27[1].m128i_i32[0] = 779382377;
      v27[1].m128i_i8[4] = 115;
      *v27 = v28;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 9:
      v29 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v29;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".max.L2::cache_hint.u";
        goto LABEL_37;
      }
      v30 = _mm_load_si128((const __m128i *)&xmmword_435F5E0);
      v29[1].m128i_i32[0] = 779382377;
      v29[1].m128i_i8[4] = 117;
      *v29 = v30;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 0xA:
      v31 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v31;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".min.L2::cache_hint.u";
        goto LABEL_37;
      }
      v32 = _mm_load_si128((const __m128i *)&xmmword_435F5F0);
      v31[1].m128i_i32[0] = 779382377;
      v31[1].m128i_i8[4] = 117;
      *v31 = v32;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 0xB:
      v33 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v33;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".add.L2::cache_hint.f";
        goto LABEL_37;
      }
      v34 = _mm_load_si128((const __m128i *)&xmmword_435F590);
      v33[1].m128i_i32[0] = 779382377;
      v33[1].m128i_i8[4] = 102;
      *v33 = v34;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 0xC:
      v23 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v23;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".inc.L2::cache_hint.u";
        goto LABEL_37;
      }
      v24 = _mm_load_si128((const __m128i *)&xmmword_435F600);
      v23[1].m128i_i32[0] = 779382377;
      v23[1].m128i_i8[4] = 117;
      *v23 = v24;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 0xD:
      v25 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v25;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".dec.L2::cache_hint.u";
        goto LABEL_37;
      }
      v26 = _mm_load_si128((const __m128i *)&xmmword_435F610);
      v25[1].m128i_i32[0] = 779382377;
      v25[1].m128i_i8[4] = 117;
      *v25 = v26;
      *(_QWORD *)(a4 + 32) += 21LL;
      break;
    case 0xE:
      v8 = *(__m128i **)(a4 + 32);
      result = *(_QWORD *)(a4 + 24) - (_QWORD)v8;
      if ( result <= 0x14 )
      {
        v35 = 21;
        v36 = ".cas.L2::cache_hint.b";
LABEL_37:
        result = sub_CB6200(a4, (unsigned __int8 *)v36, v35);
      }
      else
      {
        v9 = _mm_load_si128((const __m128i *)&xmmword_435F620);
        v8[1].m128i_i32[0] = 779382377;
        v8[1].m128i_i8[4] = 98;
        *v8 = v9;
        *(_QWORD *)(a4 + 32) += 21LL;
      }
      break;
    default:
      return result;
  }
  return result;
}
