// Function: sub_21E94F0
// Address: 0x21e94f0
//
void __fastcall sub_21E94F0(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  __int64 v5; // rax
  void *v6; // rdx
  __m128i *v7; // rdx
  void *v8; // rdx
  void *v9; // rdx

  if ( a4 && !strcmp(a4, "mode") )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8) & 0xFLL;
    if ( (_BYTE)v5 == 2 )
    {
      v9 = *(void **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v9 <= 9u )
      {
        sub_16E7EE0(a3, "membar.sys", 0xAu);
      }
      else
      {
        qmemcpy(v9, "membar.sys", 10);
        *(_QWORD *)(a3 + 24) += 10LL;
      }
    }
    else if ( (unsigned __int8)v5 > 2u )
    {
      if ( (_BYTE)v5 != 4 )
        sub_16BD130("Bad membar op", 1u);
      v7 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v7 <= 0xFu )
      {
        sub_16E7EE0(a3, "fence.sc.cluster", 0x10u);
      }
      else
      {
        *v7 = _mm_load_si128((const __m128i *)&xmmword_435F6B0);
        *(_QWORD *)(a3 + 24) += 16LL;
      }
    }
    else if ( (_BYTE)v5 )
    {
      v6 = *(void **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v6 <= 9u )
      {
        sub_16E7EE0(a3, "membar.cta", 0xAu);
      }
      else
      {
        qmemcpy(v6, "membar.cta", 10);
        *(_QWORD *)(a3 + 24) += 10LL;
      }
    }
    else
    {
      v8 = *(void **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v8 <= 9u )
      {
        sub_16E7EE0(a3, "membar.gpu", 0xAu);
      }
      else
      {
        qmemcpy(v8, "membar.gpu", 10);
        *(_QWORD *)(a3 + 24) += 10LL;
      }
    }
  }
}
