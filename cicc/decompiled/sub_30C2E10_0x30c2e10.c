// Function: sub_30C2E10
// Address: 0x30c2e10
//
__int64 __fastcall sub_30C2E10(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r12
  __int64 v7; // rdi
  _BYTE *v8; // rax
  void *v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __m128i *v13; // rdx
  __m128i v14; // xmm0
  __int64 v15; // r12
  char *v16; // rax
  size_t v17; // rdx
  _BYTE *v18; // rdi
  unsigned __int8 *v19; // rsi
  _BYTE *v20; // rax
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // r12
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // r12
  __int64 result; // rax
  __int64 i; // r13
  __int64 v29; // r14
  const char *v30; // rax
  size_t v31; // rdx
  _BYTE *v32; // rdi
  unsigned __int8 *v33; // rsi
  _BYTE *v34; // rax
  __m128i *v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // r14
  char *v38; // rax
  size_t v39; // rdx
  _BYTE *v40; // rdi
  unsigned __int8 *v41; // rsi
  _BYTE *v42; // rax
  void *v43; // rdx
  __int64 v44; // rdi
  __int64 v45; // rdi
  _BYTE *v46; // rax
  __int64 v47; // rdi
  _BYTE *v48; // rax
  __int64 v49; // rdi
  _BYTE *v50; // rax
  _BYTE *v51; // rdx
  _BYTE *v52; // rdx
  _BYTE *v53; // rax
  size_t v54; // [rsp+8h] [rbp-58h]
  size_t v55; // [rsp+8h] [rbp-58h]
  size_t v56; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v57; // [rsp+10h] [rbp-50h] BYREF
  size_t v58; // [rsp+18h] [rbp-48h]
  _QWORD v59[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x16u )
  {
    v6 = sub_CB6200(a2, "Shader Model Version : ", 0x17u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CBC60);
    v4[1].m128i_i8[6] = 32;
    v6 = a2;
    v4[1].m128i_i32[0] = 1852795251;
    v4[1].m128i_i16[2] = 14880;
    *v4 = si128;
    *(_QWORD *)(a2 + 32) += 23LL;
  }
  sub_F04FD0((__int64)&v57, (char *)(a1 + 16));
  v7 = sub_CB6200(v6, v57, v58);
  v8 = *(_BYTE **)(v7 + 32);
  if ( *(_BYTE **)(v7 + 24) == v8 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v8 = 10;
    ++*(_QWORD *)(v7 + 32);
  }
  if ( v57 != (unsigned __int8 *)v59 )
    j_j___libc_free_0((unsigned __int64)v57);
  v9 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v9 <= 0xEu )
  {
    v10 = sub_CB6200(a2, "DXIL Version : ", 0xFu);
  }
  else
  {
    v10 = a2;
    qmemcpy(v9, "DXIL Version : ", 15);
    *(_QWORD *)(a2 + 32) += 15LL;
  }
  sub_F04FD0((__int64)&v57, (char *)a1);
  v11 = sub_CB6200(v10, v57, v58);
  v12 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == v12 )
  {
    sub_CB6200(v11, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v12 = 10;
    ++*(_QWORD *)(v11 + 32);
  }
  if ( v57 != (unsigned __int8 *)v59 )
    j_j___libc_free_0((unsigned __int64)v57);
  v13 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 <= 0x15u )
  {
    v15 = sub_CB6200(a2, "Target Shader Stage : ", 0x16u);
  }
  else
  {
    v14 = _mm_load_si128((const __m128i *)&xmmword_44CBC70);
    v13[1].m128i_i32[0] = 543516513;
    v15 = a2;
    v13[1].m128i_i16[2] = 8250;
    *v13 = v14;
    *(_QWORD *)(a2 + 32) += 22LL;
  }
  v16 = sub_CC63C0(*(_DWORD *)(a1 + 32));
  v18 = *(_BYTE **)(v15 + 32);
  v19 = (unsigned __int8 *)v16;
  v20 = *(_BYTE **)(v15 + 24);
  if ( v17 > v20 - v18 )
  {
    v15 = sub_CB6200(v15, v19, v17);
    v20 = *(_BYTE **)(v15 + 24);
    v18 = *(_BYTE **)(v15 + 32);
  }
  else if ( v17 )
  {
    v56 = v17;
    memcpy(v18, v19, v17);
    v53 = *(_BYTE **)(v15 + 24);
    v18 = (_BYTE *)(*(_QWORD *)(v15 + 32) + v56);
    *(_QWORD *)(v15 + 32) = v18;
    if ( v53 != v18 )
      goto LABEL_18;
    goto LABEL_61;
  }
  if ( v20 != v18 )
  {
LABEL_18:
    *v18 = 10;
    ++*(_QWORD *)(v15 + 32);
    goto LABEL_19;
  }
LABEL_61:
  sub_CB6200(v15, (unsigned __int8 *)"\n", 1u);
LABEL_19:
  v21 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v21 <= 0x13u )
  {
    v23 = sub_CB6200(a2, "Validator Version : ", 0x14u);
  }
  else
  {
    v22 = _mm_load_si128((const __m128i *)&xmmword_44CBC80);
    v21[1].m128i_i32[0] = 540680302;
    v23 = a2;
    *v21 = v22;
    *(_QWORD *)(a2 + 32) += 20LL;
  }
  sub_F04FD0((__int64)&v57, (char *)(a1 + 36));
  v24 = sub_CB6200(v23, v57, v58);
  v25 = *(_BYTE **)(v24 + 32);
  if ( *(_BYTE **)(v24 + 24) == v25 )
  {
    sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v25 = 10;
    ++*(_QWORD *)(v24 + 32);
  }
  if ( v57 != (unsigned __int8 *)v59 )
    j_j___libc_free_0((unsigned __int64)v57);
  v26 = *(_QWORD *)(a1 + 56);
  result = 3LL * *(unsigned int *)(a1 + 64);
  for ( i = v26 + 24LL * *(unsigned int *)(a1 + 64); i != v26; result = sub_CB6200(v49, (unsigned __int8 *)"\n", 1u) )
  {
    while ( 1 )
    {
      v50 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v50 )
      {
        v29 = sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
      }
      else
      {
        *v50 = 32;
        v29 = a2;
        ++*(_QWORD *)(a2 + 32);
      }
      v30 = sub_BD5D20(*(_QWORD *)v26);
      v32 = *(_BYTE **)(v29 + 32);
      v33 = (unsigned __int8 *)v30;
      v34 = *(_BYTE **)(v29 + 24);
      if ( v31 > v34 - v32 )
      {
        v29 = sub_CB6200(v29, v33, v31);
        v34 = *(_BYTE **)(v29 + 24);
        v32 = *(_BYTE **)(v29 + 32);
      }
      else if ( v31 )
      {
        v54 = v31;
        memcpy(v32, v33, v31);
        v51 = (_BYTE *)(*(_QWORD *)(v29 + 32) + v54);
        *(_QWORD *)(v29 + 32) = v51;
        v34 = *(_BYTE **)(v29 + 24);
        v32 = v51;
      }
      if ( v34 == v32 )
      {
        sub_CB6200(v29, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v32 = 10;
        ++*(_QWORD *)(v29 + 32);
      }
      v35 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v35 <= 0x19u )
      {
        v37 = sub_CB6200(a2, "  Function Shader Stage : ", 0x1Au);
      }
      else
      {
        v36 = _mm_load_si128((const __m128i *)&xmmword_44CBC90);
        qmemcpy(&v35[1], "r Stage : ", 10);
        v37 = a2;
        *v35 = v36;
        *(_QWORD *)(a2 + 32) += 26LL;
      }
      v38 = sub_CC63C0(*(_DWORD *)(v26 + 8));
      v40 = *(_BYTE **)(v37 + 32);
      v41 = (unsigned __int8 *)v38;
      v42 = *(_BYTE **)(v37 + 24);
      if ( v42 - v40 < v39 )
      {
        v37 = sub_CB6200(v37, v41, v39);
        v42 = *(_BYTE **)(v37 + 24);
        v40 = *(_BYTE **)(v37 + 32);
      }
      else if ( v39 )
      {
        v55 = v39;
        memcpy(v40, v41, v39);
        v52 = (_BYTE *)(*(_QWORD *)(v37 + 32) + v55);
        *(_QWORD *)(v37 + 32) = v52;
        v42 = *(_BYTE **)(v37 + 24);
        v40 = v52;
      }
      if ( v42 == v40 )
      {
        sub_CB6200(v37, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v40 = 10;
        ++*(_QWORD *)(v37 + 32);
      }
      v43 = *(void **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v43 <= 0xDu )
      {
        v44 = sub_CB6200(a2, "  NumThreads: ", 0xEu);
      }
      else
      {
        v44 = a2;
        qmemcpy(v43, "  NumThreads: ", 14);
        *(_QWORD *)(a2 + 32) += 14LL;
      }
      v45 = sub_CB59D0(v44, *(unsigned int *)(v26 + 12));
      v46 = *(_BYTE **)(v45 + 32);
      if ( *(_BYTE **)(v45 + 24) == v46 )
      {
        v45 = sub_CB6200(v45, (unsigned __int8 *)",", 1u);
      }
      else
      {
        *v46 = 44;
        ++*(_QWORD *)(v45 + 32);
      }
      v47 = sub_CB59D0(v45, *(unsigned int *)(v26 + 16));
      v48 = *(_BYTE **)(v47 + 32);
      if ( *(_BYTE **)(v47 + 24) == v48 )
      {
        v47 = sub_CB6200(v47, (unsigned __int8 *)",", 1u);
      }
      else
      {
        *v48 = 44;
        ++*(_QWORD *)(v47 + 32);
      }
      v49 = sub_CB59D0(v47, *(unsigned int *)(v26 + 20));
      result = *(_QWORD *)(v49 + 32);
      if ( *(_QWORD *)(v49 + 24) == result )
        break;
      v26 += 24;
      *(_BYTE *)result = 10;
      ++*(_QWORD *)(v49 + 32);
      if ( i == v26 )
        return result;
    }
    v26 += 24;
  }
  return result;
}
