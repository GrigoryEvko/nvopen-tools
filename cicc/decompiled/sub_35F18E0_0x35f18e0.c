// Function: sub_35F18E0
// Address: 0x35f18e0
//
void __fastcall sub_35F18E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v5; // rax
  void *v6; // rdx
  __m128i *v7; // rdx
  void *v8; // rdx
  void *v9; // rdx

  if ( a5 && !strcmp(a5, "mode") )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 0xFLL;
    if ( (_BYTE)v5 == 2 )
    {
      v9 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 9u )
      {
        sub_CB6200(a4, "membar.sys", 0xAu);
      }
      else
      {
        qmemcpy(v9, "membar.sys", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
      }
    }
    else if ( (unsigned __int8)v5 > 2u )
    {
      if ( (_BYTE)v5 != 4 )
        sub_C64ED0("Bad membar op", 1u);
      v7 = *(__m128i **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v7 <= 0xFu )
      {
        sub_CB6200(a4, "fence.sc.cluster", 0x10u);
      }
      else
      {
        *v7 = _mm_load_si128((const __m128i *)&xmmword_435F6B0);
        *(_QWORD *)(a4 + 32) += 16LL;
      }
    }
    else if ( (_BYTE)v5 )
    {
      v6 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v6 <= 9u )
      {
        sub_CB6200(a4, "membar.cta", 0xAu);
      }
      else
      {
        qmemcpy(v6, "membar.cta", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
      }
    }
    else
    {
      v8 = *(void **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v8 <= 9u )
      {
        sub_CB6200(a4, "membar.gpu", 0xAu);
      }
      else
      {
        qmemcpy(v8, "membar.gpu", 10);
        *(_QWORD *)(a4 + 32) += 10LL;
      }
    }
  }
}
