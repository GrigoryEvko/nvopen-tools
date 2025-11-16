// Function: sub_30FB3B0
// Address: 0x30fb3b0
//
_BYTE *__fastcall sub_30FB3B0(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // rdi
  __m128i v13; // xmm0
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __m128i *v16; // rdx
  __m128i v17; // xmm0
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // r9
  _BYTE *v21; // rax
  const char *v22; // rax
  size_t v23; // rdx
  char *v24; // rsi
  __int64 v25; // rax
  _BYTE *v26; // rax
  __m128i *v27; // rdx
  __m128i v28; // xmm0
  __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // rdi
  _BYTE *v32; // rax
  unsigned int v33; // r12d
  __int64 v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // edx
  __int64 *v38; // rsi
  __int64 v39; // r10
  _BYTE *v40; // r9
  size_t v41; // rdx
  unsigned __int8 *v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  _WORD *v47; // rdx
  _WORD *v48; // rdi
  unsigned __int64 v49; // rax
  int v50; // esi
  const char *v51; // rax
  _BYTE *result; // rax
  __int64 v53; // rax
  _BYTE *v54; // rdx
  int v55; // r11d
  size_t v56; // [rsp+10h] [rbp-1B0h]
  size_t v57; // [rsp+18h] [rbp-1A8h]
  __int64 v58[52]; // [rsp+20h] [rbp-1A0h] BYREF

  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x18u )
  {
    v6 = sub_CB6200(a2, "[MLInlineAdvisor] Nodes: ", 0x19u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CEC60);
    v4[1].m128i_i8[8] = 32;
    v6 = a2;
    v4[1].m128i_i64[0] = 0x3A7365646F4E205DLL;
    *v4 = si128;
    *(_QWORD *)(a2 + 32) += 25LL;
  }
  v7 = sub_CB59F0(v6, *(_QWORD *)(a1 + 176));
  v8 = *(_QWORD **)(v7 + 32);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 7u )
  {
    v9 = sub_CB6200(v7, (unsigned __int8 *)" Edges: ", 8u);
  }
  else
  {
    *v8 = 0x203A736567644520LL;
    *(_QWORD *)(v7 + 32) += 8LL;
  }
  v10 = sub_CB59F0(v9, *(_QWORD *)(a1 + 184));
  v11 = *(__m128i **)(v10 + 32);
  v12 = v10;
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 0x16u )
  {
    v12 = sub_CB6200(v10, " EdgesOfLastSeenNodes: ", 0x17u);
  }
  else
  {
    v13 = _mm_load_si128((const __m128i *)&xmmword_44CEC70);
    v11[1].m128i_i32[0] = 1701080910;
    v11[1].m128i_i16[2] = 14963;
    v11[1].m128i_i8[6] = 32;
    *v11 = v13;
    *(_QWORD *)(v10 + 32) += 23LL;
  }
  v14 = sub_CB59F0(v12, *(_QWORD *)(a1 + 192));
  v15 = *(_BYTE **)(v14 + 32);
  if ( *(_BYTE **)(v14 + 24) == v15 )
  {
    sub_CB6200(v14, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v15 = 10;
    ++*(_QWORD *)(v14 + 32);
  }
  v16 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 0x16u )
  {
    sub_CB6200(a2, "[MLInlineAdvisor] FPI:\n", 0x17u);
  }
  else
  {
    v17 = _mm_load_si128((const __m128i *)&xmmword_44CEC60);
    v16[1].m128i_i32[0] = 1346773085;
    v16[1].m128i_i16[2] = 14921;
    v16[1].m128i_i8[6] = 10;
    *v16 = v17;
    *(_QWORD *)(a2 + 32) += 23LL;
  }
  v18 = *(_QWORD *)(a1 + 144);
  v19 = a1 + 128;
  if ( v18 != a1 + 128 )
  {
    while ( 1 )
    {
      qmemcpy(v58, (const void *)(v18 + 32), 0x168u);
      v22 = sub_BD5D20(v58[0]);
      v48 = *(_WORD **)(a2 + 32);
      v24 = (char *)v22;
      v49 = *(_QWORD *)(a2 + 24) - (_QWORD)v48;
      if ( v49 >= v23 )
      {
        v20 = a2;
        if ( v23 )
        {
          v56 = v23;
          memcpy(v48, v24, v23);
          v46 = *(_QWORD *)(a2 + 24);
          v47 = (_WORD *)(*(_QWORD *)(a2 + 32) + v56);
          v20 = a2;
          *(_QWORD *)(a2 + 32) = v47;
          v48 = v47;
          v49 = v46 - (_QWORD)v47;
        }
        if ( v49 <= 1 )
        {
LABEL_21:
          sub_CB6200(v20, (unsigned __int8 *)":\n", 2u);
          goto LABEL_17;
        }
      }
      else
      {
        v25 = sub_CB6200(a2, (unsigned __int8 *)v24, v23);
        v48 = *(_WORD **)(v25 + 32);
        v20 = v25;
        if ( *(_QWORD *)(v25 + 24) - (_QWORD)v48 <= 1u )
          goto LABEL_21;
      }
      *v48 = 2618;
      *(_QWORD *)(v20 + 32) += 2LL;
LABEL_17:
      sub_30C4110(&v58[1], a2);
      v21 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v21 )
      {
        sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        v18 = sub_220EEE0(v18);
        if ( v19 == v18 )
          break;
      }
      else
      {
        *v21 = 10;
        ++*(_QWORD *)(a2 + 32);
        v18 = sub_220EEE0(v18);
        if ( v19 == v18 )
          break;
      }
    }
  }
  v26 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v26 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    v27 = *(__m128i **)(a2 + 32);
  }
  else
  {
    *v26 = 10;
    v27 = (__m128i *)(*(_QWORD *)(a2 + 32) + 1LL);
    *(_QWORD *)(a2 + 32) = v27;
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v27 <= 0x1Du )
  {
    sub_CB6200(a2, "[MLInlineAdvisor] FuncLevels:\n", 0x1Eu);
  }
  else
  {
    v28 = _mm_load_si128((const __m128i *)&xmmword_44CEC60);
    qmemcpy(&v27[1], "] FuncLevels:\n", 14);
    *v27 = v28;
    *(_QWORD *)(a2 + 32) += 30LL;
  }
  v29 = *(_QWORD *)(a1 + 224);
  v30 = a1 + 208;
  if ( a1 + 208 != v29 )
  {
    while ( 1 )
    {
      v33 = *(_DWORD *)(v29 + 40);
      v34 = *(_QWORD *)(a1 + 336);
      v35 = *(_QWORD *)(*(_QWORD *)(v29 + 32) + 8LL);
      v36 = *(unsigned int *)(a1 + 352);
      if ( (_DWORD)v36 )
      {
        v37 = (v36 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v38 = (__int64 *)(v34 + 8LL * v37);
        v39 = *v38;
        if ( v35 == *v38 )
        {
LABEL_34:
          if ( v38 != (__int64 *)(v34 + 8 * v36) )
          {
            v40 = *(_BYTE **)(a2 + 32);
            v41 = 9;
            v42 = "<deleted>";
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v40 > 8u )
              goto LABEL_47;
LABEL_36:
            v43 = sub_CB6200(a2, v42, v41);
            v40 = *(_BYTE **)(v43 + 32);
            v44 = v43;
            v45 = *(_QWORD *)(v43 + 24) - (_QWORD)v40;
            goto LABEL_37;
          }
        }
        else
        {
          v50 = 1;
          while ( v39 != -4096 )
          {
            v55 = v50 + 1;
            v37 = (v36 - 1) & (v50 + v37);
            v38 = (__int64 *)(v34 + 8LL * v37);
            v39 = *v38;
            if ( v35 == *v38 )
              goto LABEL_34;
            v50 = v55;
          }
        }
      }
      v51 = sub_BD5D20(v35);
      v40 = *(_BYTE **)(a2 + 32);
      v42 = (unsigned __int8 *)v51;
      v45 = *(_QWORD *)(a2 + 24) - (_QWORD)v40;
      if ( v45 < v41 )
        goto LABEL_36;
      if ( v41 )
      {
LABEL_47:
        v57 = v41;
        memcpy(v40, v42, v41);
        v53 = *(_QWORD *)(a2 + 24);
        v44 = a2;
        v54 = (_BYTE *)(*(_QWORD *)(a2 + 32) + v57);
        *(_QWORD *)(a2 + 32) = v54;
        v40 = v54;
        v45 = v53 - (_QWORD)v54;
        goto LABEL_37;
      }
      v44 = a2;
LABEL_37:
      if ( v45 > 2 )
      {
        v40[2] = 32;
        *(_WORD *)v40 = 14880;
        *(_QWORD *)(v44 + 32) += 3LL;
      }
      else
      {
        v44 = sub_CB6200(v44, (unsigned __int8 *)" : ", 3u);
      }
      v31 = sub_CB59D0(v44, v33);
      v32 = *(_BYTE **)(v31 + 32);
      if ( *(_BYTE **)(v31 + 24) == v32 )
      {
        sub_CB6200(v31, (unsigned __int8 *)"\n", 1u);
        v29 = sub_220EF30(v29);
        if ( v30 == v29 )
          break;
      }
      else
      {
        *v32 = 10;
        ++*(_QWORD *)(v31 + 32);
        v29 = sub_220EF30(v29);
        if ( v30 == v29 )
          break;
      }
    }
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  *result = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
