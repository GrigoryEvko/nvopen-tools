// Function: sub_21E9060
// Address: 0x21e9060
//
void __fastcall sub_21E9060(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  char *v5; // rdx
  __m128i v6; // xmm0
  __m128i v7; // xmm0
  __m128i si128; // xmm0
  __m128i v9; // xmm0
  __m128i v10; // xmm0

  if ( a4 && !strcmp(a4, "name") )
  {
    v5 = *(char **)(a3 + 24);
    switch ( (unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8) )
    {
      case 0u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0x13u )
        {
          sub_16E7EE0(a3, "%is_explicit_cluster", 0x14u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_435F640);
          *((_DWORD *)v5 + 4) = 1919251571;
          *(__m128i *)v5 = si128;
          *(_QWORD *)(a3 + 24) += 20LL;
        }
        break;
      case 1u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xFu )
        {
          sub_16E7EE0(a3, "%cluster_ctarank", 0x10u);
        }
        else
        {
          *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_435F650);
          *(_QWORD *)(a3 + 24) += 16LL;
        }
        break;
      case 2u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0x10u )
        {
          sub_16E7EE0(a3, "%cluster_nctarank", 0x11u);
        }
        else
        {
          v6 = _mm_load_si128((const __m128i *)&xmmword_435F660);
          v5[16] = 107;
          *(__m128i *)v5 = v6;
          *(_QWORD *)(a3 + 24) += 17LL;
        }
        break;
      case 3u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0x10u )
        {
          sub_16E7EE0(a3, "%cluster_nctaid.x", 0x11u);
        }
        else
        {
          v10 = _mm_load_si128((const __m128i *)&xmmword_435F670);
          v5[16] = 120;
          *(__m128i *)v5 = v10;
          *(_QWORD *)(a3 + 24) += 17LL;
        }
        break;
      case 4u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0x10u )
        {
          sub_16E7EE0(a3, "%cluster_nctaid.y", 0x11u);
        }
        else
        {
          v7 = _mm_load_si128((const __m128i *)&xmmword_435F670);
          v5[16] = 121;
          *(__m128i *)v5 = v7;
          *(_QWORD *)(a3 + 24) += 17LL;
        }
        break;
      case 5u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0x10u )
        {
          sub_16E7EE0(a3, "%cluster_nctaid.z", 0x11u);
        }
        else
        {
          v9 = _mm_load_si128((const __m128i *)&xmmword_435F670);
          v5[16] = 122;
          *(__m128i *)v5 = v9;
          *(_QWORD *)(a3 + 24) += 17LL;
        }
        break;
      case 6u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xFu )
        {
          sub_16E7EE0(a3, "%cluster_ctaid.x", 0x10u);
        }
        else
        {
          *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_435F680);
          *(_QWORD *)(a3 + 24) += 16LL;
        }
        break;
      case 7u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xFu )
        {
          sub_16E7EE0(a3, "%cluster_ctaid.y", 0x10u);
        }
        else
        {
          *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_435F690);
          *(_QWORD *)(a3 + 24) += 16LL;
        }
        break;
      case 8u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xFu )
        {
          sub_16E7EE0(a3, "%cluster_ctaid.z", 0x10u);
        }
        else
        {
          *(__m128i *)v5 = _mm_load_si128((const __m128i *)&xmmword_435F6A0);
          *(_QWORD *)(a3 + 24) += 16LL;
        }
        break;
      case 9u:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xCu )
        {
          sub_16E7EE0(a3, "%nclusterid.x", 0xDu);
        }
        else
        {
          qmemcpy(v5, "%nclusterid.x", 13);
          *(_QWORD *)(a3 + 24) += 13LL;
        }
        break;
      case 0xAu:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xCu )
        {
          sub_16E7EE0(a3, "%nclusterid.y", 0xDu);
        }
        else
        {
          qmemcpy(v5, "%nclusterid.y", 13);
          *(_QWORD *)(a3 + 24) += 13LL;
        }
        break;
      case 0xBu:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xCu )
        {
          sub_16E7EE0(a3, "%nclusterid.z", 0xDu);
        }
        else
        {
          qmemcpy(v5, "%nclusterid.z", 13);
          *(_QWORD *)(a3 + 24) += 13LL;
        }
        break;
      case 0xCu:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xBu )
        {
          sub_16E7EE0(a3, "%clusterid.x", 0xCu);
        }
        else
        {
          qmemcpy(v5, "%clusterid.x", 12);
          *(_QWORD *)(a3 + 24) += 12LL;
        }
        break;
      case 0xDu:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xBu )
        {
          sub_16E7EE0(a3, "%clusterid.y", 0xCu);
        }
        else
        {
          qmemcpy(v5, "%clusterid.y", 12);
          *(_QWORD *)(a3 + 24) += 12LL;
        }
        break;
      case 0xEu:
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 0xBu )
        {
          sub_16E7EE0(a3, "%clusterid.z", 0xCu);
        }
        else
        {
          qmemcpy(v5, "%clusterid.z", 12);
          *(_QWORD *)(a3 + 24) += 12LL;
        }
        break;
      default:
        sub_16BD130("Unhandled cluster info operand", 1u);
    }
  }
}
