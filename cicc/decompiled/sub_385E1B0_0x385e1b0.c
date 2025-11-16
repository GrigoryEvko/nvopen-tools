// Function: sub_385E1B0
// Address: 0x385e1b0
//
__int64 __fastcall sub_385E1B0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax
  __m128i *v7; // rdx
  __m128i v8; // xmm0
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rdx
  _WORD *v20; // rdx
  unsigned int v21; // ebx
  __int64 i; // r13
  _BYTE *v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // r15
  unsigned int v28; // [rsp+10h] [rbp-40h]

  v3 = sub_16E8750(a2, a3);
  v4 = *(__m128i **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0x17u )
  {
    sub_16E7EE0(v3, "Run-time memory checks:\n", 0x18u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F717D0);
    v4[1].m128i_i64[0] = 0xA3A736B63656863LL;
    *v4 = si128;
    *(_QWORD *)(v3 + 24) += 24LL;
  }
  sub_385DDE0(a1, a2, a1 + 272, a3);
  v6 = sub_16E8750(a2, a3);
  v7 = *(__m128i **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0x11u )
  {
    sub_16E7EE0(v6, "Grouped accesses:\n", 0x12u);
  }
  else
  {
    v8 = _mm_load_si128((const __m128i *)&xmmword_3F717E0);
    v7[1].m128i_i16[0] = 2618;
    *v7 = v8;
    *(_QWORD *)(v6 + 24) += 18LL;
  }
  result = a1;
  if ( *(_DWORD *)(a1 + 160) )
  {
    v28 = 0;
    result = 0;
    do
    {
      v10 = *(_QWORD *)(a1 + 152) + 48 * result;
      v11 = sub_16E8750(a2, a3 + 2);
      v12 = *(_QWORD *)(v11 + 24);
      v13 = v11;
      if ( (unsigned __int64)(*(_QWORD *)(v11 + 16) - v12) <= 5 )
      {
        v13 = sub_16E7EE0(v11, "Group ", 6u);
      }
      else
      {
        *(_DWORD *)v12 = 1970238023;
        *(_WORD *)(v12 + 4) = 8304;
        *(_QWORD *)(v11 + 24) += 6LL;
      }
      v14 = sub_16E7B40(v13, v10);
      v15 = *(_WORD **)(v14 + 24);
      if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
      {
        sub_16E7EE0(v14, ":\n", 2u);
      }
      else
      {
        *v15 = 2618;
        *(_QWORD *)(v14 + 24) += 2LL;
      }
      v16 = sub_16E8750(a2, a3 + 4);
      v17 = *(_QWORD *)(v16 + 24);
      v18 = v16;
      if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v17) <= 5 )
      {
        v18 = sub_16E7EE0(v16, "(Low: ", 6u);
      }
      else
      {
        *(_DWORD *)v17 = 2003782696;
        *(_WORD *)(v17 + 4) = 8250;
        *(_QWORD *)(v16 + 24) += 6LL;
      }
      sub_1456620(*(_QWORD *)(v10 + 16), v18);
      v19 = *(_QWORD *)(v18 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v18 + 16) - v19) <= 6 )
      {
        v18 = sub_16E7EE0(v18, " High: ", 7u);
      }
      else
      {
        *(_DWORD *)v19 = 1734952992;
        *(_WORD *)(v19 + 4) = 14952;
        *(_BYTE *)(v19 + 6) = 32;
        *(_QWORD *)(v18 + 24) += 7LL;
      }
      sub_1456620(*(_QWORD *)(v10 + 8), v18);
      v20 = *(_WORD **)(v18 + 24);
      if ( *(_QWORD *)(v18 + 16) - (_QWORD)v20 <= 1u )
      {
        sub_16E7EE0(v18, ")\n", 2u);
      }
      else
      {
        *v20 = 2601;
        *(_QWORD *)(v18 + 24) += 2LL;
      }
      v21 = 0;
      for ( i = 0; v21 < *(_DWORD *)(v10 + 32); i = v21 )
      {
        while ( 1 )
        {
          v24 = sub_16E8750(a2, a3 + 6);
          v25 = *(_QWORD **)(v24 + 24);
          v26 = v24;
          if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 > 7u )
          {
            *v25 = 0x203A7265626D654DLL;
            *(_QWORD *)(v24 + 24) += 8LL;
          }
          else
          {
            v26 = sub_16E7EE0(v24, "Member: ", 8u);
          }
          sub_1456620(
            *(_QWORD *)(*(_QWORD *)(a1 + 8)
                      + ((unsigned __int64)*(unsigned int *)(*(_QWORD *)(v10 + 24) + 4 * i) << 6)
                      + 56),
            v26);
          v23 = *(_BYTE **)(v26 + 24);
          if ( *(_BYTE **)(v26 + 16) == v23 )
            break;
          i = v21 + 1;
          *v23 = 10;
          v21 = i;
          ++*(_QWORD *)(v26 + 24);
          if ( (unsigned int)i >= *(_DWORD *)(v10 + 32) )
            goto LABEL_25;
        }
        sub_16E7EE0(v26, "\n", 1u);
        ++v21;
      }
LABEL_25:
      result = ++v28;
    }
    while ( v28 < *(_DWORD *)(a1 + 160) );
  }
  return result;
}
