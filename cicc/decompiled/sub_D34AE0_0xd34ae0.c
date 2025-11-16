// Function: sub_D34AE0
// Address: 0xd34ae0
//
unsigned __int64 __fastcall sub_D34AE0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax
  __m128i *v7; // rdx
  __m128i v8; // xmm0
  unsigned __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rdx
  _WORD *v19; // rdx
  unsigned int *v20; // r12
  _BYTE *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // r15
  unsigned __int64 i; // [rsp+8h] [rbp-58h]
  unsigned __int64 v28; // [rsp+18h] [rbp-48h]
  unsigned int *v30; // [rsp+28h] [rbp-38h]

  v3 = sub_CB69B0(a2, a3);
  v4 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x17u )
  {
    sub_CB6200(v3, "Run-time memory checks:\n", 0x18u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F717D0);
    v4[1].m128i_i64[0] = 0xA3A736B63656863LL;
    *v4 = si128;
    *(_QWORD *)(v3 + 32) += 24LL;
  }
  sub_D34730(a1, a2, a1 + 296, a3);
  v6 = sub_CB69B0(a2, a3);
  v7 = *(__m128i **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0x11u )
  {
    sub_CB6200(v6, "Grouped accesses:\n", 0x12u);
  }
  else
  {
    v8 = _mm_load_si128((const __m128i *)&xmmword_3F717E0);
    v7[1].m128i_i16[0] = 2618;
    *v7 = v8;
    *(_QWORD *)(v6 + 32) += 18LL;
  }
  v28 = *(_QWORD *)(a1 + 168);
  result = v28 + 48LL * *(unsigned int *)(a1 + 176);
  for ( i = result; i != v28; result = v28 )
  {
    v10 = sub_CB69B0(a2, a3 + 2);
    v11 = *(_QWORD *)(v10 + 32);
    v12 = v10;
    if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v11) <= 5 )
    {
      v12 = sub_CB6200(v10, "Group ", 6u);
    }
    else
    {
      *(_DWORD *)v11 = 1970238023;
      *(_WORD *)(v11 + 4) = 8304;
      *(_QWORD *)(v10 + 32) += 6LL;
    }
    v13 = sub_CB5A80(v12, v28);
    v14 = *(_WORD **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 1u )
    {
      sub_CB6200(v13, (unsigned __int8 *)":\n", 2u);
    }
    else
    {
      *v14 = 2618;
      *(_QWORD *)(v13 + 32) += 2LL;
    }
    v15 = sub_CB69B0(a2, a3 + 4);
    v16 = *(_QWORD *)(v15 + 32);
    v17 = v15;
    if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 5 )
    {
      v17 = sub_CB6200(v15, "(Low: ", 6u);
    }
    else
    {
      *(_DWORD *)v16 = 2003782696;
      *(_WORD *)(v16 + 4) = 8250;
      *(_QWORD *)(v15 + 32) += 6LL;
    }
    sub_D955C0(*(_QWORD *)(v28 + 8), v17);
    v18 = *(_QWORD *)(v17 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v18) <= 6 )
    {
      v17 = sub_CB6200(v17, " High: ", 7u);
    }
    else
    {
      *(_DWORD *)v18 = 1734952992;
      *(_WORD *)(v18 + 4) = 14952;
      *(_BYTE *)(v18 + 6) = 32;
      *(_QWORD *)(v17 + 32) += 7LL;
    }
    sub_D955C0(*(_QWORD *)v28, v17);
    v19 = *(_WORD **)(v17 + 32);
    if ( *(_QWORD *)(v17 + 24) - (_QWORD)v19 <= 1u )
    {
      sub_CB6200(v17, (unsigned __int8 *)")\n", 2u);
    }
    else
    {
      *v19 = 2601;
      *(_QWORD *)(v17 + 32) += 2LL;
    }
    v20 = *(unsigned int **)(v28 + 16);
    v30 = &v20[*(unsigned int *)(v28 + 24)];
    while ( v30 != v20 )
    {
      while ( 1 )
      {
        v22 = *v20;
        v23 = sub_CB69B0(a2, a3 + 6);
        v24 = *(_QWORD **)(v23 + 32);
        v25 = v23;
        if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 > 7u )
        {
          *v24 = 0x203A7265626D654DLL;
          *(_QWORD *)(v23 + 32) += 8LL;
        }
        else
        {
          v25 = sub_CB6200(v23, "Member: ", 8u);
        }
        sub_D955C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 72 * v22 + 56), v25);
        v21 = *(_BYTE **)(v25 + 32);
        if ( *(_BYTE **)(v25 + 24) == v21 )
          break;
        *v21 = 10;
        ++v20;
        ++*(_QWORD *)(v25 + 32);
        if ( v30 == v20 )
          goto LABEL_24;
      }
      ++v20;
      sub_CB6200(v25, (unsigned __int8 *)"\n", 1u);
    }
LABEL_24:
    v28 += 48LL;
  }
  return result;
}
