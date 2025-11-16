// Function: sub_2C7AF00
// Address: 0x2c7af00
//
void __fastcall sub_2C7AF00(__int64 a1, char *a2)
{
  __int64 v2; // r14
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rax
  unsigned __int16 v9; // cx
  __m128i *v10; // rdx
  __int64 v11; // r13
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rax
  size_t v16; // rdx
  _BYTE *v17; // rdi
  char *v18; // rsi
  _BYTE *v19; // rax
  char *v20; // rcx
  _WORD *v21; // rdx
  _BYTE *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdx
  char *v25; // rsi
  __int64 v26; // rax
  char *v27; // rcx
  __m128i *v28; // rdx
  __int64 v29; // rdi
  __m128i si128; // xmm0
  __int64 v31; // rdx
  __m128i v32; // xmm0
  _BYTE *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  __m128i v37; // xmm0
  _BYTE *v38; // rax
  __int64 v39; // rax
  __m128i *v40; // rdx
  __int64 v41; // rdi
  __m128i v42; // xmm0
  __int64 v43; // rsi
  __int64 v44; // rdx
  __m128i v45; // xmm0
  __int64 v46; // rcx
  _BYTE *v47; // rax
  __int64 v48; // rdi
  unsigned __int8 v49; // al
  unsigned __int8 v50; // al
  int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rax
  char *v56; // rcx
  __m128i *v57; // rdx
  __int64 v58; // rdi
  void *v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  _BYTE *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int16 v67; // [rsp+8h] [rbp-68h]
  unsigned __int16 v68; // [rsp+8h] [rbp-68h]
  size_t v69; // [rsp+8h] [rbp-68h]
  _QWORD v70[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v71[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v72; // [rsp+30h] [rbp-40h] BYREF
  __int64 v73; // [rsp+38h] [rbp-38h]

  v2 = (__int64)a2;
  v4 = *((_QWORD *)a2 + 1);
  if ( *(_BYTE *)(v4 + 8) == 12 )
  {
    v70[0] = sub_BCAE30(*((_QWORD *)a2 + 1));
    v70[1] = v52;
    if ( sub_CA1930(v70) != 32 )
    {
      v71[0] = sub_BCAE30(v4);
      v71[1] = v53;
      if ( sub_CA1930(v71) != 64 )
      {
        v72 = sub_BCAE30(v4);
        v73 = v54;
        if ( sub_CA1930(&v72) != 128 )
        {
          v55 = sub_2C76A00(a1, (__int64)a2, 0);
          v57 = *(__m128i **)(v55 + 32);
          v58 = v55;
          if ( *(_QWORD *)(v55 + 24) - (_QWORD)v57 <= 0x2Fu )
          {
            a2 = "Atomic operations on non-i32/i64/i128 types are ";
            v66 = sub_CB6200(v55, "Atomic operations on non-i32/i64/i128 types are ", 0x30u);
            v59 = *(void **)(v66 + 32);
            v58 = v66;
          }
          else
          {
            *v57 = _mm_load_si128((const __m128i *)&xmmword_42D0AF0);
            v57[1] = _mm_load_si128((const __m128i *)&xmmword_42D0B00);
            v57[2] = _mm_load_si128((const __m128i *)&xmmword_42D0B10);
            v59 = (void *)(*(_QWORD *)(v55 + 32) + 48LL);
            *(_QWORD *)(v55 + 32) = v59;
          }
          if ( *(_QWORD *)(v58 + 24) - (_QWORD)v59 <= 0xDu )
          {
            a2 = "not supported\n";
            sub_CB6200(v58, (unsigned __int8 *)"not supported\n", 0xEu);
          }
          else
          {
            qmemcpy(v59, "not supported\n", 14);
            *(_QWORD *)(v58 + 32) += 14LL;
          }
          sub_2C76240(a1, (__int64)a2, (__int64)v59, v56);
        }
      }
    }
  }
  v5 = *(_QWORD *)(v2 - 64);
  v6 = *(_QWORD *)(v5 + 8);
  if ( *(_BYTE *)(v6 + 8) == 14 )
  {
    v7 = *(_DWORD *)(v6 + 8);
    if ( v7 <= 0x1FF || v7 >> 8 == 3 )
      goto LABEL_4;
    v25 = (char *)v2;
    v26 = sub_2C76A00(a1, v2, 0);
    v28 = *(__m128i **)(v26 + 32);
    v29 = v26;
    if ( *(_QWORD *)(v26 + 24) - (_QWORD)v28 <= 0x27u )
    {
      v25 = "atomicrmw pointer operand must point to ";
      v64 = sub_CB6200(v26, "atomicrmw pointer operand must point to ", 0x28u);
      v31 = *(_QWORD *)(v64 + 32);
      v29 = v64;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42D0D80);
      v28[2].m128i_i64[0] = 0x206F7420746E696FLL;
      *v28 = si128;
      v28[1] = _mm_load_si128((const __m128i *)&xmmword_42D0DA0);
      v31 = *(_QWORD *)(v26 + 32) + 40LL;
      *(_QWORD *)(v26 + 32) = v31;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v31) <= 0x28 )
    {
      v25 = "generic, global, or shared address space\n";
      sub_CB6200(v29, "generic, global, or shared address space\n", 0x29u);
    }
    else
    {
      v32 = _mm_load_si128((const __m128i *)&xmmword_42D0DB0);
      *(_BYTE *)(v31 + 40) = 10;
      *(_QWORD *)(v31 + 32) = 0x6563617073207373LL;
      *(__m128i *)v31 = v32;
      *(__m128i *)(v31 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0DC0);
      *(_QWORD *)(v29 + 32) += 41LL;
    }
  }
  else
  {
    v25 = (char *)v2;
    v35 = sub_2C76A00(a1, v2, 0);
    v31 = *(_QWORD *)(v35 + 32);
    v36 = v35;
    if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v31) <= 0x27 )
    {
      v25 = "atomicrmw pointer operand not a pointer?";
      v36 = sub_CB6200(v35, "atomicrmw pointer operand not a pointer?", 0x28u);
      v38 = *(_BYTE **)(v36 + 32);
    }
    else
    {
      v37 = _mm_load_si128((const __m128i *)&xmmword_42D0D80);
      *(_QWORD *)(v31 + 32) = 0x3F7265746E696F70LL;
      *(__m128i *)v31 = v37;
      *(__m128i *)(v31 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0D90);
      v38 = (_BYTE *)(*(_QWORD *)(v35 + 32) + 40LL);
      *(_QWORD *)(v36 + 32) = v38;
    }
    if ( v38 == *(_BYTE **)(v36 + 24) )
    {
      v25 = "\n";
      sub_CB6200(v36, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v38 = 10;
      ++*(_QWORD *)(v36 + 32);
    }
  }
  v33 = *(_BYTE **)(a1 + 16);
  if ( v33 )
    *v33 = 0;
  if ( !*(_DWORD *)(a1 + 4) )
  {
    v34 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(v34 + 32) != *(_QWORD *)(v34 + 16) )
    {
      sub_CB5AE0((__int64 *)v34);
      v34 = *(_QWORD *)(a1 + 24);
    }
    sub_CEB520(*(_QWORD **)(v34 + 48), (__int64)v25, v31, v27);
  }
LABEL_4:
  if ( ((*(_WORD *)(v2 + 2) >> 4) & 0x1F) == 0xB )
  {
    v49 = *(_BYTE *)(v4 + 8);
    if ( *(_DWORD *)a1 == 1 )
    {
      if ( (!v49 || v49 == 3) && *(_DWORD *)(*(_QWORD *)(v5 + 8) + 8LL) >> 8 == 1 )
        goto LABEL_23;
    }
    else if ( (unsigned int)v49 - 17 > 1 )
    {
      if ( v49 <= 3u )
        goto LABEL_23;
    }
    else
    {
      v50 = *(_BYTE *)(*(_QWORD *)(v4 + 24) + 8LL);
      if ( v50 == 2 )
      {
        if ( ((*(_DWORD *)(v4 + 32) - 2) & 0xFFFFFFFD) == 0 )
          goto LABEL_23;
      }
      else if ( v50 <= 1u )
      {
        v51 = *(_DWORD *)(v4 + 32);
        if ( ((v51 - 2) & 0xFFFFFFFD) == 0 || v51 == 8 )
          goto LABEL_23;
      }
    }
  }
  else if ( ((*(_WORD *)(v2 + 2) >> 4) & 0x1Fu) <= 0xB )
  {
    if ( ((*(_WORD *)(v2 + 2) >> 4) & 0x1F) != 0xB )
      goto LABEL_23;
  }
  else if ( (unsigned __int16)(((*(_WORD *)(v2 + 2) >> 4) & 0x1F) - 12) <= 4u )
  {
    goto LABEL_23;
  }
  v67 = (*(_WORD *)(v2 + 2) >> 4) & 0x1F;
  v8 = sub_2C76A00(a1, v2, 0);
  v9 = v67;
  v10 = *(__m128i **)(v8 + 32);
  v11 = v8;
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v10 <= 0x23u )
  {
    v60 = sub_CB6200(v8, "atomicrmw does not support operation", 0x24u);
    v9 = v67;
    v11 = v60;
    v13 = *(_QWORD *)(v60 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v60 + 24) - v13) > 2 )
      goto LABEL_9;
LABEL_69:
    v68 = v9;
    v61 = sub_CB6200(v11, (unsigned __int8 *)": '", 3u);
    v9 = v68;
    v11 = v61;
    goto LABEL_10;
  }
  v12 = _mm_load_si128((const __m128i *)&xmmword_42D0DD0);
  v10[2].m128i_i32[0] = 1852795252;
  *v10 = v12;
  v10[1] = _mm_load_si128((const __m128i *)&xmmword_42D0DE0);
  v13 = *(_QWORD *)(v8 + 32) + 36LL;
  v14 = *(_QWORD *)(v8 + 24);
  *(_QWORD *)(v11 + 32) = v13;
  if ( (unsigned __int64)(v14 - v13) <= 2 )
    goto LABEL_69;
LABEL_9:
  *(_BYTE *)(v13 + 2) = 39;
  *(_WORD *)v13 = 8250;
  *(_QWORD *)(v11 + 32) += 3LL;
LABEL_10:
  v15 = sub_B4D7D0(v9);
  v17 = *(_BYTE **)(v11 + 32);
  v18 = v15;
  v19 = *(_BYTE **)(v11 + 24);
  v20 = (char *)(v19 - v17);
  if ( v19 - v17 < v16 )
  {
    v11 = sub_CB6200(v11, (unsigned __int8 *)v18, v16);
    v19 = *(_BYTE **)(v11 + 24);
    v17 = *(_BYTE **)(v11 + 32);
  }
  else if ( v16 )
  {
    v69 = v16;
    memcpy(v17, v18, v16);
    v63 = (_BYTE *)(*(_QWORD *)(v11 + 32) + v69);
    *(_QWORD *)(v11 + 32) = v63;
    v19 = *(_BYTE **)(v11 + 24);
    v17 = v63;
  }
  if ( v17 == v19 )
  {
    v18 = "'";
    v62 = sub_CB6200(v11, (unsigned __int8 *)"'", 1u);
    v21 = *(_WORD **)(v62 + 32);
    v11 = v62;
  }
  else
  {
    *v17 = 39;
    v21 = (_WORD *)(*(_QWORD *)(v11 + 32) + 1LL);
    *(_QWORD *)(v11 + 32) = v21;
  }
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v21 <= 1u )
  {
    v18 = ".\n";
    sub_CB6200(v11, (unsigned __int8 *)".\n", 2u);
  }
  else
  {
    *v21 = 2606;
    *(_QWORD *)(v11 + 32) += 2LL;
  }
  v22 = *(_BYTE **)(a1 + 16);
  if ( v22 )
    *v22 = 0;
  if ( !*(_DWORD *)(a1 + 4) )
  {
    v23 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(v23 + 32) != *(_QWORD *)(v23 + 16) )
    {
      sub_CB5AE0((__int64 *)v23);
      v23 = *(_QWORD *)(a1 + 24);
    }
    sub_CEB520(*(_QWORD **)(v23 + 48), (__int64)v18, (__int64)v21, v20);
  }
LABEL_23:
  v72 = sub_BCAE30(v4);
  v73 = v24;
  if ( sub_CA1930(&v72) == 128 && (*(_WORD *)(v2 + 2) & 0x1F0) != 0 )
  {
    v39 = sub_2C76A00(a1, v2, 0);
    v40 = *(__m128i **)(v39 + 32);
    v41 = v39;
    if ( *(_QWORD *)(v39 + 24) - (_QWORD)v40 <= 0x32u )
    {
      v43 = (__int64)"Atomic operations on i128 types are only supported ";
      v65 = sub_CB6200(v39, "Atomic operations on i128 types are only supported ", 0x33u);
      v44 = *(_QWORD *)(v65 + 32);
      v41 = v65;
    }
    else
    {
      v42 = _mm_load_si128((const __m128i *)&xmmword_42D0AF0);
      v43 = 25701;
      v40[3].m128i_i8[2] = 32;
      v40[3].m128i_i16[0] = 25701;
      *v40 = v42;
      v40[1] = _mm_load_si128((const __m128i *)&xmmword_42D0DF0);
      v40[2] = _mm_load_si128((const __m128i *)&xmmword_42D0E00);
      v44 = *(_QWORD *)(v39 + 32) + 51LL;
      *(_QWORD *)(v39 + 32) = v44;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v41 + 24) - v44) <= 0x12 )
    {
      v43 = (__int64)"for xchg operation\n";
      sub_CB6200(v41, "for xchg operation\n", 0x13u);
    }
    else
    {
      v45 = _mm_load_si128((const __m128i *)&xmmword_42D0E10);
      v46 = 28271;
      *(_BYTE *)(v44 + 18) = 10;
      *(_WORD *)(v44 + 16) = 28271;
      *(__m128i *)v44 = v45;
      *(_QWORD *)(v41 + 32) += 19LL;
    }
    v47 = *(_BYTE **)(a1 + 16);
    if ( v47 )
      *v47 = 0;
    if ( !*(_DWORD *)(a1 + 4) )
    {
      v48 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v48 + 32) != *(_QWORD *)(v48 + 16) )
      {
        sub_CB5AE0((__int64 *)v48);
        v48 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v48 + 48), v43, v44, (char *)v46);
    }
  }
  sub_2C795F0(a1, v2);
}
