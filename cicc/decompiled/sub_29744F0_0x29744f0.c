// Function: sub_29744F0
// Address: 0x29744f0
//
_BYTE *__fastcall sub_29744F0(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __m128i v19; // xmm0
  __int64 v20; // rax
  __int64 v21; // rdi
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rdi
  __m128i v25; // xmm0
  _BYTE *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdi
  __m128i v30; // xmm0
  __m128i *v31; // rax
  __int64 v32; // rdi
  __m128i v33; // xmm0
  __int64 v34; // rax
  __int64 v35; // rdi
  __m128i v36; // xmm0
  __int64 v37; // rax
  __int64 v38; // rdi
  __m128i v39; // xmm0
  __int64 v40; // rax
  __int64 v41; // rdi
  __m128i v42; // xmm0
  __int64 v43; // rax
  __int64 v44; // rdi
  __m128i v45; // xmm0
  __int64 v46; // rax
  __int64 v47; // rdi
  __m128i v48; // xmm0
  _BYTE *result; // rax
  unsigned __int64 v50; // rax

  v6 = a3(a4, "SimplifyCFGPass]", 15);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v50 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v50 > (unsigned __int64)v8 )
      goto LABEL_4;
    goto LABEL_67;
  }
  if ( v10 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_67:
  sub_CB5D20(a2, 60);
LABEL_5:
  v12 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v12 <= 0x14u )
  {
    v14 = sub_CB6200(a2, "bonus-inst-threshold=", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4399370);
    v12[1].m128i_i32[0] = 1684828008;
    v14 = a2;
    v12[1].m128i_i8[4] = 61;
    *v12 = si128;
    *(_QWORD *)(a2 + 32) += 21LL;
  }
  v15 = sub_CB59F0(v14, *(int *)a1);
  v16 = *(_BYTE **)(v15 + 32);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
  {
    sub_CB5D20(v15, 59);
  }
  else
  {
    *(_QWORD *)(v15 + 32) = v16 + 1;
    *v16 = 59;
  }
  v17 = *(_QWORD *)(a2 + 32);
  v18 = a2;
  if ( !a1[4] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v17) > 2 )
    {
      *(_BYTE *)(v17 + 2) = 45;
      *(_WORD *)v17 = 28526;
      v17 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v17;
    }
    else
    {
      v18 = sub_CB6200(a2, "no-", 3u);
      v17 = *(_QWORD *)(v18 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v18 + 24) - v17) <= 0x13 )
  {
    sub_CB6200(v18, "forward-switch-cond;", 0x14u);
  }
  else
  {
    v19 = _mm_load_si128((const __m128i *)&xmmword_4399380);
    *(_DWORD *)(v17 + 16) = 996437615;
    *(__m128i *)v17 = v19;
    *(_QWORD *)(v18 + 32) += 20LL;
  }
  v20 = *(_QWORD *)(a2 + 32);
  v21 = a2;
  if ( !a1[5] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v20) > 2 )
    {
      *(_BYTE *)(v20 + 2) = 45;
      *(_WORD *)v20 = 28526;
      v20 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v20;
    }
    else
    {
      v21 = sub_CB6200(a2, "no-", 3u);
      v20 = *(_QWORD *)(v21 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v20) <= 0x14 )
  {
    sub_CB6200(v21, "switch-range-to-icmp;", 0x15u);
  }
  else
  {
    v22 = _mm_load_si128((const __m128i *)&xmmword_4399390);
    *(_DWORD *)(v20 + 16) = 1886217065;
    *(_BYTE *)(v20 + 20) = 59;
    *(__m128i *)v20 = v22;
    *(_QWORD *)(v21 + 32) += 21LL;
  }
  v23 = *(_QWORD *)(a2 + 32);
  v24 = a2;
  if ( !a1[6] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v23) > 2 )
    {
      *(_BYTE *)(v23 + 2) = 45;
      *(_WORD *)v23 = 28526;
      v23 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v23;
    }
    else
    {
      v24 = sub_CB6200(a2, "no-", 3u);
      v23 = *(_QWORD *)(v24 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v23) <= 0x10 )
  {
    sub_CB6200(v24, "switch-to-lookup;", 0x11u);
  }
  else
  {
    v25 = _mm_load_si128((const __m128i *)&xmmword_43993A0);
    *(_BYTE *)(v23 + 16) = 59;
    *(__m128i *)v23 = v25;
    *(_QWORD *)(v24 + 32) += 17LL;
  }
  v26 = *(_BYTE **)(a2 + 32);
  v27 = a2;
  if ( !a1[7] )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 > 2u )
    {
      v26[2] = 45;
      *(_WORD *)v26 = 28526;
      v26 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v26;
    }
    else
    {
      v27 = sub_CB6200(a2, "no-", 3u);
      v26 = *(_BYTE **)(v27 + 32);
    }
  }
  if ( *(_QWORD *)(v27 + 24) - (_QWORD)v26 <= 0xAu )
  {
    sub_CB6200(v27, "keep-loops;", 0xBu);
  }
  else
  {
    qmemcpy(v26, "keep-loops;", 11);
    *(_QWORD *)(v27 + 32) += 11LL;
  }
  v28 = *(_QWORD *)(a2 + 32);
  v29 = a2;
  if ( !a1[8] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v28) > 2 )
    {
      *(_BYTE *)(v28 + 2) = 45;
      *(_WORD *)v28 = 28526;
      v28 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v28;
    }
    else
    {
      v29 = sub_CB6200(a2, "no-", 3u);
      v28 = *(_QWORD *)(v29 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v28) <= 0x12 )
  {
    sub_CB6200(v29, "hoist-common-insts;", 0x13u);
  }
  else
  {
    v30 = _mm_load_si128((const __m128i *)&xmmword_43993B0);
    *(_BYTE *)(v28 + 18) = 59;
    *(_WORD *)(v28 + 16) = 29556;
    *(__m128i *)v28 = v30;
    *(_QWORD *)(v29 + 32) += 19LL;
  }
  v31 = *(__m128i **)(a2 + 32);
  v32 = a2;
  if ( !a1[9] )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v31 > 2u )
    {
      v31->m128i_i8[2] = 45;
      v31->m128i_i16[0] = 28526;
      v31 = (__m128i *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v31;
    }
    else
    {
      v32 = sub_CB6200(a2, "no-", 3u);
      v31 = *(__m128i **)(v32 + 32);
    }
  }
  if ( *(_QWORD *)(v32 + 24) - (_QWORD)v31 <= 0x25u )
  {
    sub_CB6200(v32, "hoist-loads-stores-with-cond-faulting;", 0x26u);
  }
  else
  {
    v33 = _mm_load_si128((const __m128i *)&xmmword_43993C0);
    v31[2].m128i_i32[0] = 1852404844;
    v31[2].m128i_i16[2] = 15207;
    *v31 = v33;
    v31[1] = _mm_load_si128((const __m128i *)&xmmword_43993D0);
    *(_QWORD *)(v32 + 32) += 38LL;
  }
  v34 = *(_QWORD *)(a2 + 32);
  v35 = a2;
  if ( !a1[10] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v34) > 2 )
    {
      *(_BYTE *)(v34 + 2) = 45;
      *(_WORD *)v34 = 28526;
      v34 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v34;
    }
    else
    {
      v35 = sub_CB6200(a2, "no-", 3u);
      v34 = *(_QWORD *)(v35 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v34) <= 0x11 )
  {
    sub_CB6200(v35, "sink-common-insts;", 0x12u);
  }
  else
  {
    v36 = _mm_load_si128((const __m128i *)&xmmword_43993E0);
    *(_WORD *)(v34 + 16) = 15219;
    *(__m128i *)v34 = v36;
    *(_QWORD *)(v35 + 32) += 18LL;
  }
  v37 = *(_QWORD *)(a2 + 32);
  v38 = a2;
  if ( !a1[14] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v37) > 2 )
    {
      *(_BYTE *)(v37 + 2) = 45;
      *(_WORD *)v37 = 28526;
      v37 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v37;
    }
    else
    {
      v38 = sub_CB6200(a2, "no-", 3u);
      v37 = *(_QWORD *)(v38 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v37) <= 0x10 )
  {
    sub_CB6200(v38, "speculate-blocks;", 0x11u);
  }
  else
  {
    v39 = _mm_load_si128((const __m128i *)&xmmword_43993F0);
    *(_BYTE *)(v37 + 16) = 59;
    *(__m128i *)v37 = v39;
    *(_QWORD *)(v38 + 32) += 17LL;
  }
  v40 = *(_QWORD *)(a2 + 32);
  v41 = a2;
  if ( !a1[11] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v40) > 2 )
    {
      *(_BYTE *)(v40 + 2) = 45;
      *(_WORD *)v40 = 28526;
      v40 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v40;
    }
    else
    {
      v41 = sub_CB6200(a2, "no-", 3u);
      v40 = *(_QWORD *)(v41 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v41 + 24) - v40) <= 0x14 )
  {
    sub_CB6200(v41, "simplify-cond-branch;", 0x15u);
  }
  else
  {
    v42 = _mm_load_si128((const __m128i *)&xmmword_4399400);
    *(_DWORD *)(v40 + 16) = 1751346785;
    *(_BYTE *)(v40 + 20) = 59;
    *(__m128i *)v40 = v42;
    *(_QWORD *)(v41 + 32) += 21LL;
  }
  v43 = *(_QWORD *)(a2 + 32);
  v44 = a2;
  if ( !a1[12] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v43) > 2 )
    {
      *(_BYTE *)(v43 + 2) = 45;
      *(_WORD *)v43 = 28526;
      v43 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v43;
    }
    else
    {
      v44 = sub_CB6200(a2, "no-", 3u);
      v43 = *(_QWORD *)(v44 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v44 + 24) - v43) <= 0x14 )
  {
    sub_CB6200(v44, "simplify-unreachable;", 0x15u);
  }
  else
  {
    v45 = _mm_load_si128((const __m128i *)&xmmword_4399410);
    *(_DWORD *)(v43 + 16) = 1701601889;
    *(_BYTE *)(v43 + 20) = 59;
    *(__m128i *)v43 = v45;
    *(_QWORD *)(v44 + 32) += 21LL;
  }
  v46 = *(_QWORD *)(a2 + 32);
  v47 = a2;
  if ( !a1[15] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v46) > 2 )
    {
      *(_BYTE *)(v46 + 2) = 45;
      *(_WORD *)v46 = 28526;
      v46 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v46;
    }
    else
    {
      v47 = sub_CB6200(a2, "no-", 3u);
      v46 = *(_QWORD *)(v47 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v47 + 24) - v46) <= 0x17 )
  {
    sub_CB6200(v47, "speculate-unpredictables", 0x18u);
  }
  else
  {
    v48 = _mm_load_si128((const __m128i *)&xmmword_4399420);
    *(_QWORD *)(v46 + 16) = 0x73656C6261746369LL;
    *(__m128i *)v46 = v48;
    *(_QWORD *)(v47 + 32) += 24LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
