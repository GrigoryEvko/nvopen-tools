// Function: sub_2287290
// Address: 0x2287290
//
__int64 __fastcall sub_2287290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned __int64 i; // rbx
  __int64 v5; // rdi
  __int64 v6; // r8
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __m128i si128; // xmm0
  __int64 v11; // rdi
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __m128i *v19; // rdx
  unsigned __int8 v20; // bl
  unsigned __int64 v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __m128i v24; // xmm0
  unsigned __int8 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v29; // [rsp+20h] [rbp-70h] BYREF
  size_t v30; // [rsp+28h] [rbp-68h]
  _QWORD v31[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int8 *v32[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+50h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a3 + 16);
  v28 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)(a1 + 16) )
  {
    v23 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v23) <= 8 )
    {
      sub_CB6200(a2, "</tr><tr>", 9u);
    }
    else
    {
      *(_BYTE *)(v23 + 8) = 62;
      *(_QWORD *)v23 = 0x72743C3E72742F3CLL;
      *(_QWORD *)(a2 + 32) += 9LL;
    }
  }
  if ( v28 == v3 )
    return 0;
  v26 = 0;
  for ( i = 0; i != 64; ++i )
  {
    v29 = (unsigned __int8 *)v31;
    sub_2285B20((__int64 *)&v29, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v30 )
    {
      v8 = *(_QWORD *)(a2 + 24);
      v9 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(a1 + 16) )
      {
        if ( (unsigned __int64)(v8 - v9) <= 0x16 )
        {
          v11 = sub_CB6200(a2, "<td colspan=\"1\" port=\"s", 0x17u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
          *(_DWORD *)(v9 + 16) = 1953656688;
          *(_WORD *)(v9 + 20) = 8765;
          v11 = a2;
          *(_BYTE *)(v9 + 22) = 115;
          *(__m128i *)v9 = si128;
          *(_QWORD *)(a2 + 32) += 23LL;
        }
        v12 = sub_CB59D0(v11, i);
        v13 = *(_WORD **)(v12 + 32);
        v14 = v12;
        if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
        {
          v14 = sub_CB6200(v12, "\">", 2u);
        }
        else
        {
          *v13 = 15906;
          *(_QWORD *)(v12 + 32) += 2LL;
        }
        v15 = sub_CB6200(v14, v29, v30);
        v16 = *(_QWORD *)(v15 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v16) <= 4 )
        {
          sub_CB6200(v15, "</td>", 5u);
        }
        else
        {
          *(_DWORD *)v16 = 1685335868;
          *(_BYTE *)(v16 + 4) = 62;
          *(_QWORD *)(v15 + 32) += 5LL;
        }
LABEL_11:
        if ( v29 != (unsigned __int8 *)v31 )
          j_j___libc_free_0((unsigned __int64)v29);
        v26 = 1;
LABEL_14:
        v3 += 40;
        if ( v28 == v3 )
          return v26;
        continue;
      }
      if ( i )
      {
        if ( v8 != v9 )
        {
          *(_BYTE *)v9 = 124;
          v9 = *(_QWORD *)(a2 + 32) + 1LL;
          v17 = *(_QWORD *)(a2 + 24);
          *(_QWORD *)(a2 + 32) = v9;
          if ( (unsigned __int64)(v17 - v9) > 1 )
            goto LABEL_6;
          goto LABEL_26;
        }
        sub_CB6200(a2, (unsigned __int8 *)"|", 1u);
        v9 = *(_QWORD *)(a2 + 32);
        v8 = *(_QWORD *)(a2 + 24);
      }
      if ( (unsigned __int64)(v8 - v9) > 1 )
      {
LABEL_6:
        v5 = a2;
        *(_WORD *)v9 = 29500;
        *(_QWORD *)(a2 + 32) += 2LL;
        goto LABEL_7;
      }
LABEL_26:
      v5 = sub_CB6200(a2, "<s", 2u);
LABEL_7:
      v6 = sub_CB59D0(v5, i);
      v7 = *(_BYTE **)(v6 + 32);
      if ( *(_BYTE **)(v6 + 24) == v7 )
      {
        v6 = sub_CB6200(v6, (unsigned __int8 *)">", 1u);
      }
      else
      {
        *v7 = 62;
        ++*(_QWORD *)(v6 + 32);
      }
      v27 = v6;
      sub_C67200((__int64 *)v32, (__int64)&v29);
      sub_CB6200(v27, v32[0], (size_t)v32[1]);
      if ( (__int64 *)v32[0] != &v33 )
        j_j___libc_free_0((unsigned __int64)v32[0]);
      goto LABEL_11;
    }
    if ( v29 == (unsigned __int8 *)v31 )
      goto LABEL_14;
    v3 += 40;
    j_j___libc_free_0((unsigned __int64)v29);
    if ( v28 == v3 )
      return v26;
  }
  if ( v26 )
  {
    v19 = *(__m128i **)(a2 + 32);
    v20 = *(_BYTE *)(a1 + 16);
    v21 = *(_QWORD *)(a2 + 24) - (_QWORD)v19;
    if ( v20 )
    {
      if ( v21 <= 0x2B )
      {
        sub_CB6200(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>", 0x2Cu);
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
        qmemcpy(&v19[2], "ated...</td>", 12);
        *v19 = v22;
        v19[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB40);
        *(_QWORD *)(a2 + 32) += 44LL;
      }
      return v20;
    }
    else if ( v21 <= 0x11 )
    {
      sub_CB6200(a2, "|<s64>truncated...", 0x12u);
    }
    else
    {
      v24 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
      v19[1].m128i_i16[0] = 11822;
      *v19 = v24;
      *(_QWORD *)(a2 + 32) += 18LL;
    }
  }
  return v26;
}
