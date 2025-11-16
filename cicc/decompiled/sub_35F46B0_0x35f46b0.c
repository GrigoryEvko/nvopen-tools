// Function: sub_35F46B0
// Address: 0x35f46b0
//
void __fastcall sub_35F46B0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rax
  __m128i *v10; // rdx
  __m128i si128; // xmm0
  _BYTE *v12; // rdx
  unsigned __int64 v13; // rax
  void *v14; // rdx
  _QWORD *v15; // rdx
  char *v16; // rsi
  __m128i *v17; // rdx
  _QWORD *v18; // rdx

  if ( a5 )
  {
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
      v14 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v14 <= 0xBu )
      {
        sub_CB6200(a4, (unsigned __int8 *)".shared::cta", 0xCu);
      }
      else
      {
        qmemcpy(v14, ".shared::cta", 12);
        *(_QWORD *)(a4 + 32) += 12LL;
      }
    }
    if ( *(_BYTE *)a5 == 111 && *(_BYTE *)(a5 + 1) == 112 && !*(_BYTE *)(a5 + 2) )
    {
      v12 = *(_BYTE **)(a4 + 32);
      v13 = *(_QWORD *)(a4 + 24) - (_QWORD)v12;
      if ( (v7 & 0xF0) != 0x10 )
      {
        if ( v13 <= 9 )
        {
          sub_CB6200(a4, ".test_wait", 0xAu);
        }
        else
        {
          qmemcpy(v12, ".test_wait", 10);
          *(_QWORD *)(a4 + 32) += 10LL;
        }
        if ( strcmp((const char *)a5, "parity_op") )
          goto LABEL_12;
LABEL_10:
        v10 = *(__m128i **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 <= 0x10u )
        {
          sub_CB6200(a4, ".test_wait.parity", 0x11u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44FE860);
          v10[1].m128i_i8[0] = 121;
          *v10 = si128;
          *(_QWORD *)(a4 + 32) += 17LL;
        }
        goto LABEL_12;
      }
      if ( v13 <= 8 )
      {
        sub_CB6200(a4, ".try_wait", 9u);
      }
      else
      {
        v12[8] = 116;
        *(_QWORD *)v12 = 0x6961775F7972742ELL;
        *(_QWORD *)(a4 + 32) += 9LL;
      }
      if ( strcmp((const char *)a5, "parity_op") )
      {
LABEL_12:
        if ( !strcmp((const char *)a5, "sem_ordered") )
        {
          v15 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v15 > 7u )
          {
            *v15 = 0x657269757163612ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return;
          }
          v16 = ".acquire";
        }
        else
        {
          if ( strcmp((const char *)a5, "sem_unordered") )
            return;
          v18 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v18 > 7u )
          {
            *v18 = 0x646578616C65722ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return;
          }
          v16 = ".relaxed";
        }
        sub_CB6200(a4, (unsigned __int8 *)v16, 8u);
        return;
      }
    }
    else
    {
      if ( strcmp((const char *)a5, "parity_op") )
        goto LABEL_12;
      if ( (v7 & 0xF0) != 0x10 )
        goto LABEL_10;
    }
    v17 = *(__m128i **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 <= 0xFu )
    {
      sub_CB6200(a4, ".try_wait.parity", 0x10u);
    }
    else
    {
      *v17 = _mm_load_si128((const __m128i *)&xmmword_44FE850);
      *(_QWORD *)(a4 + 32) += 16LL;
    }
    goto LABEL_12;
  }
}
