// Function: sub_140A360
// Address: 0x140a360
//
__int64 __fastcall sub_140A360(__int64 a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0
  _QWORD *v4; // rbx
  __int64 result; // rax
  __int64 v6; // r13
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rax
  char *v11; // rcx
  unsigned __int64 v12; // rdx
  _WORD *v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  _QWORD *v16; // rdx
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v2 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 0x22u )
  {
    sub_16E7EE0(a2, "The following are dereferenceable:\n", 35);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428B1C0);
    v2[2].m128i_i8[2] = 10;
    v2[2].m128i_i16[0] = 14949;
    *v2 = si128;
    v2[1] = _mm_load_si128((const __m128i *)&xmmword_428B1D0);
    *(_QWORD *)(a2 + 24) += 35LL;
  }
  v4 = *(_QWORD **)(a1 + 160);
  result = (__int64)&v4[*(unsigned int *)(a1 + 168)];
  v17 = (_QWORD *)result;
  if ( (_QWORD *)result != v4 )
  {
    while ( 1 )
    {
      v6 = *v4;
      sub_155C2B0(*v4, a2, 0);
      v7 = *(_QWORD **)(a1 + 224);
      v8 = *(_QWORD **)(a1 + 216);
      if ( v7 == v8 )
      {
        v9 = &v8[*(unsigned int *)(a1 + 236)];
        if ( v8 == v9 )
        {
          v16 = *(_QWORD **)(a1 + 216);
        }
        else
        {
          do
          {
            if ( v6 == *v8 )
              break;
            ++v8;
          }
          while ( v9 != v8 );
          v16 = v9;
        }
        goto LABEL_25;
      }
      v9 = &v7[*(unsigned int *)(a1 + 232)];
      v8 = (_QWORD *)sub_16CC9F0(a1 + 208, v6);
      if ( v6 == *v8 )
        break;
      v10 = *(_QWORD *)(a1 + 224);
      if ( v10 == *(_QWORD *)(a1 + 216) )
      {
        v8 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 236));
        v16 = v8;
LABEL_25:
        while ( v16 != v8 && *v8 >= 0xFFFFFFFFFFFFFFFELL )
          ++v8;
        goto LABEL_10;
      }
      v8 = (_QWORD *)(v10 + 8LL * *(unsigned int *)(a1 + 232));
LABEL_10:
      v11 = *(char **)(a2 + 24);
      v12 = *(_QWORD *)(a2 + 16) - (_QWORD)v11;
      if ( v8 == v9 )
      {
        if ( v12 <= 0xB )
        {
          sub_16E7EE0(a2, "\t(unaligned)", 12);
          v13 = *(_WORD **)(a2 + 24);
        }
        else
        {
          qmemcpy(v11, "\t(unaligned)", 12);
          v13 = (_WORD *)(*(_QWORD *)(a2 + 24) + 12LL);
          *(_QWORD *)(a2 + 24) = v13;
        }
      }
      else if ( v12 <= 9 )
      {
        sub_16E7EE0(a2, "\t(aligned)", 10);
        v13 = *(_WORD **)(a2 + 24);
      }
      else
      {
        qmemcpy(v11, "\t(aligned)", 10);
        v13 = (_WORD *)(*(_QWORD *)(a2 + 24) + 10LL);
        *(_QWORD *)(a2 + 24) = v13;
      }
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v13 > 1u )
      {
        result = 2570;
        ++v4;
        *v13 = 2570;
        *(_QWORD *)(a2 + 24) += 2LL;
        if ( v17 == v4 )
          return result;
      }
      else
      {
        ++v4;
        result = sub_16E7EE0(a2, "\n\n", 2);
        if ( v17 == v4 )
          return result;
      }
    }
    v14 = *(_QWORD *)(a1 + 224);
    if ( v14 == *(_QWORD *)(a1 + 216) )
      v15 = *(unsigned int *)(a1 + 236);
    else
      v15 = *(unsigned int *)(a1 + 232);
    v16 = (_QWORD *)(v14 + 8 * v15);
    goto LABEL_25;
  }
  return result;
}
