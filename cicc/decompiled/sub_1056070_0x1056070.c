// Function: sub_1056070
// Address: 0x1056070
//
__int64 __fastcall sub_1056070(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  const char *v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rdi
  unsigned __int8 *v8; // rsi
  unsigned __int64 v9; // rax
  _BYTE *v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  size_t v14; // [rsp+8h] [rbp-18h]

  v2 = a2;
  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x1Cu )
  {
    v2 = sub_CB6200(a2, "UniformityInfo for function '", 0x1Du);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E870);
    qmemcpy(&v3[1], "or function '", 13);
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 29LL;
  }
  v5 = sub_BD5D20(*(_QWORD *)(a1 + 176));
  v7 = *(_BYTE **)(v2 + 32);
  v8 = (unsigned __int8 *)v5;
  v9 = *(_QWORD *)(v2 + 24) - (_QWORD)v7;
  if ( v9 < v6 )
  {
    v13 = sub_CB6200(v2, v8, v6);
    v7 = *(_BYTE **)(v13 + 32);
    v2 = v13;
    v9 = *(_QWORD *)(v13 + 24) - (_QWORD)v7;
  }
  else if ( v6 )
  {
    v14 = v6;
    memcpy(v7, v8, v6);
    v11 = (_BYTE *)(*(_QWORD *)(v2 + 32) + v14);
    v12 = *(_QWORD *)(v2 + 24) - (_QWORD)v11;
    *(_QWORD *)(v2 + 32) = v11;
    v7 = v11;
    if ( v12 > 2 )
      goto LABEL_6;
    return sub_CB6200(v2, "':\n", 3u);
  }
  if ( v9 > 2 )
  {
LABEL_6:
    v7[2] = 10;
    *(_WORD *)v7 = 14887;
    *(_QWORD *)(v2 + 32) += 3LL;
    return 14887;
  }
  return sub_CB6200(v2, "':\n", 3u);
}
