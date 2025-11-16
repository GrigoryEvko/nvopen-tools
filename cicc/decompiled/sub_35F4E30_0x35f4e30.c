// Function: sub_35F4E30
// Address: 0x35f4e30
//
void __fastcall sub_35F4E30(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __m128i *v8; // rdx
  __m128i v9; // xmm0
  bool v10; // zf
  __int64 v11; // rax
  void *v12; // rdx
  __m128i *v13; // rdx
  __m128i *v14; // rdx
  __m128i si128; // xmm0

  if ( a5 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
    if ( !strcmp((const char *)a5, "group") )
    {
      v10 = (v7 & 1) == 0;
      v11 = *(_QWORD *)(a4 + 24);
      v12 = *(void **)(a4 + 32);
      if ( v10 )
      {
        if ( (unsigned __int64)(v11 - (_QWORD)v12) <= 0xC )
        {
          sub_CB6200(a4, ".cta_group::1", 0xDu);
        }
        else
        {
          qmemcpy(v12, ".cta_group::1", 13);
          *(_QWORD *)(a4 + 32) += 13LL;
        }
      }
      else if ( (unsigned __int64)(v11 - (_QWORD)v12) <= 0xC )
      {
        sub_CB6200(a4, ".cta_group::2", 0xDu);
      }
      else
      {
        qmemcpy(v12, ".cta_group::2", 13);
        *(_QWORD *)(a4 + 32) += 13LL;
      }
    }
    if ( !strcmp((const char *)a5, "arrive") )
    {
      v14 = *(__m128i **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v14 <= 0x15u )
      {
        sub_CB6200(a4, ".mbarrier::arrive::one", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44FE8A0);
        v14[1].m128i_i32[0] = 1866087013;
        v14[1].m128i_i16[2] = 25966;
        *v14 = si128;
        *(_QWORD *)(a4 + 32) += 22LL;
      }
    }
    if ( !strcmp((const char *)a5, "shared") )
    {
      v13 = *(__m128i **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v13 <= 0xFu )
      {
        sub_CB6200(a4, (unsigned __int8 *)".shared::cluster", 0x10u);
      }
      else
      {
        *v13 = _mm_load_si128((const __m128i *)&xmmword_44FE820);
        *(_QWORD *)(a4 + 32) += 16LL;
      }
    }
    if ( *(_BYTE *)a5 == 109 && *(_BYTE *)(a5 + 1) == 99 && !*(_BYTE *)(a5 + 2) )
    {
      v8 = *(__m128i **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v8 <= 0x12u )
      {
        sub_CB6200(a4, ".multicast::cluster", 0x13u);
      }
      else
      {
        v9 = _mm_load_si128((const __m128i *)&xmmword_44FE8B0);
        v8[1].m128i_i8[2] = 114;
        v8[1].m128i_i16[0] = 25972;
        *v8 = v9;
        *(_QWORD *)(a4 + 32) += 19LL;
      }
    }
  }
}
