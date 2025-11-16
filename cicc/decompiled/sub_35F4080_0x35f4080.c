// Function: sub_35F4080
// Address: 0x35f4080
//
void __fastcall sub_35F4080(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // r13
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rax
  __m128i *v10; // rcx
  unsigned __int64 v11; // rax
  unsigned int v12; // r13d
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // r15
  _BYTE *v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // eax
  _BYTE *v20; // rax
  size_t v21; // rdx
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __m128i si128; // xmm0
  unsigned __int64 v25; // [rsp-60h] [rbp-60h] BYREF
  _QWORD *v26; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v27; // [rsp-50h] [rbp-50h]
  _QWORD v28[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a5 )
    return;
  v5 = a4;
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !strcmp((const char *)a5, "scope") )
  {
    v8 = *(_QWORD **)(a4 + 32);
    v9 = *(_QWORD *)(a4 + 24) - (_QWORD)v8;
    if ( (v7 & 0xF) == 1 )
    {
      if ( v9 <= 7 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".cluster", 8u);
      }
      else
      {
        *v8 = 0x72657473756C632ELL;
        *(_QWORD *)(a4 + 32) += 8LL;
      }
    }
    else if ( v9 <= 3 )
    {
      sub_CB6200(a4, (unsigned __int8 *)".cta", 4u);
    }
    else
    {
      *(_DWORD *)v8 = 1635017518;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  if ( !strcmp((const char *)a5, "shared") )
  {
    v10 = *(__m128i **)(v5 + 32);
    v11 = *(_QWORD *)(v5 + 24) - (_QWORD)v10;
    if ( (BYTE1(v7) & 0xF0) == 0x10 )
    {
      if ( v11 <= 0xF )
      {
        sub_CB6200(v5, (unsigned __int8 *)".shared::cluster", 0x10u);
      }
      else
      {
        *v10 = _mm_load_si128((const __m128i *)&xmmword_44FE820);
        *(_QWORD *)(v5 + 32) += 16LL;
      }
    }
    else if ( v11 <= 0xB )
    {
      sub_CB6200(v5, (unsigned __int8 *)".shared::cta", 0xCu);
    }
    else
    {
      qmemcpy(v10, ".shared::cta", 12);
      *(_QWORD *)(v5 + 32) += 12LL;
    }
  }
  if ( *(_BYTE *)a5 == 111 && *(_BYTE *)(a5 + 1) == 112 && !*(_BYTE *)(a5 + 2) )
  {
    v15 = *(_BYTE **)(v5 + 32);
    if ( *(_BYTE **)(v5 + 24) == v15 )
    {
      v16 = sub_CB6200(v5, (unsigned __int8 *)".", 1u);
    }
    else
    {
      *v15 = 46;
      v16 = v5;
      ++*(_QWORD *)(v5 + 32);
    }
    switch ( (unsigned __int8)((unsigned int)v7 >> 4) )
    {
      case 0u:
        v21 = 6;
        v26 = v28;
        v22 = (unsigned __int8 *)v28;
        strcpy((char *)v28, "arrive");
        v27 = 6;
        goto LABEL_51;
      case 1u:
        v21 = 11;
        v22 = (unsigned __int8 *)v28;
        v26 = v28;
        strcpy((char *)v28, "arrive_drop");
        v27 = 11;
        goto LABEL_51;
      case 2u:
        v25 = 16;
        v26 = v28;
        v26 = (_QWORD *)sub_22409D0((__int64)&v26, &v25, 0);
        v28[0] = v25;
        *(__m128i *)v26 = _mm_load_si128((const __m128i *)&xmmword_44FE830);
        goto LABEL_54;
      case 3u:
        v25 = 21;
        v26 = v28;
        v23 = sub_22409D0((__int64)&v26, &v25, 0);
        si128 = _mm_load_si128((const __m128i *)&xmmword_44FE840);
        v26 = (_QWORD *)v23;
        v28[0] = v25;
        *(_DWORD *)(v23 + 16) = 1952412771;
        *(_BYTE *)(v23 + 20) = 120;
        *(__m128i *)v23 = si128;
LABEL_54:
        v27 = v25;
        *((_BYTE *)v26 + v25) = 0;
        v21 = v27;
        v22 = (unsigned __int8 *)v26;
        goto LABEL_51;
      case 4u:
        v21 = 9;
        v26 = v28;
        v22 = (unsigned __int8 *)v28;
        strcpy((char *)v28, "expect_tx");
        v27 = 9;
        goto LABEL_51;
      case 5u:
        v26 = v28;
        v22 = (unsigned __int8 *)v28;
        v21 = 11;
        strcpy((char *)v28, "complete_tx");
        v27 = 11;
LABEL_51:
        sub_CB6200(v16, v22, v21);
        if ( v26 != v28 )
          j_j___libc_free_0((unsigned __int64)v26);
        break;
      default:
        goto LABEL_61;
    }
  }
  if ( !strcmp((const char *)a5, "sem_ordered") )
  {
    v17 = *(_BYTE **)(v5 + 32);
    if ( *(_BYTE **)(v5 + 24) == v17 )
    {
      v18 = sub_CB6200(v5, (unsigned __int8 *)".", 1u);
    }
    else
    {
      *v17 = 46;
      v18 = v5;
      ++*(_QWORD *)(v5 + 32);
    }
    v19 = (unsigned int)v7 >> 4;
    if ( (unsigned __int8)((unsigned int)v7 >> 4) <= 3u )
    {
      v26 = v28;
      strcpy((char *)v28, "release");
      v27 = 7;
      goto LABEL_28;
    }
  }
  else
  {
    if ( strcmp((const char *)a5, "sem_unordered") )
      goto LABEL_13;
    v20 = *(_BYTE **)(v5 + 32);
    if ( *(_BYTE **)(v5 + 24) == v20 )
    {
      v18 = sub_CB6200(v5, (unsigned __int8 *)".", 1u);
    }
    else
    {
      *v20 = 46;
      v18 = v5;
      ++*(_QWORD *)(v5 + 32);
    }
    v19 = (unsigned int)v7 >> 4;
    if ( (unsigned __int8)((unsigned int)v7 >> 4) <= 3u )
      goto LABEL_33;
  }
  if ( (unsigned __int8)(v19 - 4) > 1u )
    goto LABEL_61;
LABEL_33:
  v26 = v28;
  strcpy((char *)v28, "relaxed");
  v27 = 7;
LABEL_28:
  sub_CB6200(v18, (unsigned __int8 *)v28, 7u);
  if ( v26 != v28 )
    j_j___libc_free_0((unsigned __int64)v26);
LABEL_13:
  if ( strcmp((const char *)a5, "sink") )
    return;
  v12 = (unsigned int)v7 >> 4;
  if ( (unsigned __int8)v12 > 3u )
  {
    if ( (unsigned __int8)(v12 - 4) <= 1u )
      return;
LABEL_61:
    BUG();
  }
  v13 = *(_BYTE **)(v5 + 32);
  if ( *(_BYTE **)(v5 + 24) == v13 )
  {
    v5 = sub_CB6200(v5, (unsigned __int8 *)"_", 1u);
    v14 = *(_BYTE **)(v5 + 32);
  }
  else
  {
    *v13 = 95;
    v14 = (_BYTE *)(*(_QWORD *)(v5 + 32) + 1LL);
    *(_QWORD *)(v5 + 32) = v14;
  }
  if ( *(_BYTE **)(v5 + 24) == v14 )
  {
    sub_CB6200(v5, (unsigned __int8 *)",", 1u);
  }
  else
  {
    *v14 = 44;
    ++*(_QWORD *)(v5 + 32);
  }
}
