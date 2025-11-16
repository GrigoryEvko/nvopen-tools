// Function: sub_35F38B0
// Address: 0x35f38b0
//
char __fastcall sub_35F38B0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v6; // rbx
  char **v7; // rax
  char v8; // bl
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  bool v11; // zf
  void *v12; // rdx
  __int64 v13; // rdx
  char *v14; // r14
  size_t v15; // rax
  void *v16; // rdi
  size_t v17; // r13
  __int64 v18; // rdx
  __m128i *v19; // rdx
  __m128i v20; // xmm0
  __m128i *v21; // rdx
  __m128i si128; // xmm0
  size_t v23; // rdx
  char *v24; // rsi

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  v11 = strcmp(a5, "ashift") == 0;
  LOBYTE(v7) = !v11;
  if ( v11 )
  {
    if ( (v6 & 4) == 0 )
      return (char)v7;
    v18 = *(_QWORD *)(a4 + 32);
    v7 = (char **)(*(_QWORD *)(a4 + 24) - v18);
    if ( (unsigned __int64)v7 > 6 )
    {
      *(_DWORD *)v18 = 1752391982;
      *(_WORD *)(v18 + 4) = 26217;
      *(_BYTE *)(v18 + 6) = 116;
      *(_QWORD *)(a4 + 32) += 7LL;
      return (char)v7;
    }
    v23 = 7;
    v24 = ".ashift";
LABEL_26:
    LOBYTE(v7) = sub_CB6200(a4, (unsigned __int8 *)v24, v23);
    return (char)v7;
  }
  v11 = strcmp(a5, "buffera") == 0;
  LOBYTE(v7) = !v11;
  if ( v11 )
  {
    v8 = v6 & 3;
    switch ( v8 )
    {
      case 2:
        v21 = *(__m128i **)(a4 + 32);
        v7 = (char **)(*(_QWORD *)(a4 + 24) - (_QWORD)v21);
        if ( (unsigned __int64)v7 > 0x12 )
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44FE800);
          v21[1].m128i_i8[2] = 108;
          v21[1].m128i_i16[0] = 27753;
          *v21 = si128;
          *(_QWORD *)(a4 + 32) += 19LL;
          return (char)v7;
        }
        v23 = 19;
        v24 = ".collector::a::fill";
        goto LABEL_26;
      case 3:
        v19 = *(__m128i **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 > 0x11u )
        {
          v20 = _mm_load_si128((const __m128i *)&xmmword_44FE810);
          v19[1].m128i_i16[0] = 25971;
          *v19 = v20;
          *(_QWORD *)(a4 + 32) += 18LL;
          LOBYTE(v7) = 115;
          return (char)v7;
        }
        v23 = 18;
        v24 = ".collector::a::use";
        goto LABEL_26;
      case 0:
        return (char)v7;
    }
    v9 = *(__m128i **)(a4 + 32);
    v7 = (char **)(*(_QWORD *)(a4 + 24) - (_QWORD)v9);
    if ( (unsigned __int64)v7 <= 0x15 )
    {
      v23 = 22;
      v24 = ".collector::a::lastuse";
      goto LABEL_26;
    }
    v10 = _mm_load_si128((const __m128i *)&xmmword_44FE7F0);
    v9[1].m128i_i32[0] = 1970565985;
    v9[1].m128i_i16[2] = 25971;
    *v9 = v10;
    *(_QWORD *)(a4 + 32) += 22LL;
  }
  else
  {
    v11 = memcmp(a5, "bufferb", 7u) == 0;
    LOBYTE(v7) = !v11;
    if ( v11 )
    {
      v11 = strcmp(a5, "bufferb_ws") == 0;
      LOBYTE(v7) = !v11;
      if ( (((unsigned int)v6 >> 11) & 3) != 0 || v11 )
      {
        v12 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v12 <= 0xCu )
        {
          sub_CB6200(a4, ".collector::b", 0xDu);
        }
        else
        {
          qmemcpy(v12, ".collector::b", 13);
          *(_QWORD *)(a4 + 32) += 13LL;
        }
        sub_CB59D0(a4, ((unsigned int)v6 >> 13) & 3);
        v7 = &off_49D8E20;
        v13 = ((unsigned int)v6 >> 11) & 3;
        v14 = (&off_49D8E20)[v13];
        if ( v14 )
        {
          v15 = strlen((&off_49D8E20)[v13]);
          v16 = *(void **)(a4 + 32);
          v17 = v15;
          v7 = (char **)(*(_QWORD *)(a4 + 24) - (_QWORD)v16);
          if ( v17 > (unsigned __int64)v7 )
          {
            v23 = v17;
            v24 = v14;
            goto LABEL_26;
          }
          if ( v17 )
          {
            LOBYTE(v7) = (unsigned __int8)memcpy(v16, v14, v17);
            *(_QWORD *)(a4 + 32) += v17;
          }
        }
      }
    }
  }
  return (char)v7;
}
