// Function: sub_FE1BC0
// Address: 0xfe1bc0
//
__int64 __fastcall sub_FE1BC0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rdx
  unsigned __int64 v4; // rax
  int v5; // ebx
  __int64 v6; // rax
  unsigned int v7; // r15d
  unsigned __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // r15
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __m128i si128; // xmm0
  __int64 v15; // rdi
  __int64 v16; // rax
  _WORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __m128i *v23; // rdx
  unsigned int v24; // ebx
  unsigned __int64 v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rdx
  __m128i v28; // xmm0
  __int64 v30; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v31; // [rsp+10h] [rbp-70h] BYREF
  size_t v32; // [rsp+18h] [rbp-68h]
  _QWORD v33[2]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v34[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v35; // [rsp+40h] [rbp-40h] BYREF

  v3 = (_QWORD *)(a3 + 48);
  v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v4 == v3 )
    goto LABEL_43;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
  {
LABEL_43:
    v5 = 0;
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
  }
  else
  {
    v5 = sub_B46E30(v4 - 24);
    if ( !*(_BYTE *)(a1 + 16) )
      goto LABEL_5;
  }
  v27 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v27) <= 8 )
  {
    sub_CB6200(a2, "</tr><tr>", 9u);
  }
  else
  {
    *(_BYTE *)(v27 + 8) = 62;
    *(_QWORD *)v27 = 0x72743C3E72742F3CLL;
    *(_QWORD *)(a2 + 32) += 9LL;
  }
LABEL_5:
  if ( !v5 )
    return 0;
  v6 = (unsigned int)(v5 - 1);
  v7 = 0;
  v8 = 0;
  v30 = v6;
  do
  {
    v31 = (unsigned __int8 *)v33;
    sub_FDB1F0((__int64 *)&v31, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v32 )
    {
      v12 = *(_QWORD *)(a2 + 24);
      v13 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(a1 + 16) )
      {
        if ( (unsigned __int64)(v12 - v13) <= 0x16 )
        {
          v15 = sub_CB6200(a2, "<td colspan=\"1\" port=\"s", 0x17u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
          *(_DWORD *)(v13 + 16) = 1953656688;
          *(_WORD *)(v13 + 20) = 8765;
          v15 = a2;
          *(_BYTE *)(v13 + 22) = 115;
          *(__m128i *)v13 = si128;
          *(_QWORD *)(a2 + 32) += 23LL;
        }
        v16 = sub_CB59D0(v15, v8);
        v17 = *(_WORD **)(v16 + 32);
        v18 = v16;
        if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 1u )
        {
          v18 = sub_CB6200(v16, "\">", 2u);
        }
        else
        {
          *v17 = 15906;
          *(_QWORD *)(v16 + 32) += 2LL;
        }
        v19 = sub_CB6200(v18, v31, v32);
        v20 = *(_QWORD *)(v19 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v20) <= 4 )
        {
          sub_CB6200(v19, "</td>", 5u);
        }
        else
        {
          *(_DWORD *)v20 = 1685335868;
          *(_BYTE *)(v20 + 4) = 62;
          *(_QWORD *)(v19 + 32) += 5LL;
        }
LABEL_14:
        if ( v31 != (unsigned __int8 *)v33 )
          j_j___libc_free_0(v31, v33[0] + 1LL);
        v7 = 1;
LABEL_17:
        if ( v30 == v8 )
          return v7;
        goto LABEL_18;
      }
      if ( (_DWORD)v8 )
      {
        if ( v13 != v12 )
        {
          *(_BYTE *)v13 = 124;
          v13 = *(_QWORD *)(a2 + 32) + 1LL;
          v21 = *(_QWORD *)(a2 + 24);
          *(_QWORD *)(a2 + 32) = v13;
          if ( (unsigned __int64)(v21 - v13) > 1 )
            goto LABEL_9;
          goto LABEL_29;
        }
        sub_CB6200(a2, (unsigned __int8 *)"|", 1u);
        v13 = *(_QWORD *)(a2 + 32);
        v12 = *(_QWORD *)(a2 + 24);
      }
      if ( (unsigned __int64)(v12 - v13) > 1 )
      {
LABEL_9:
        v9 = a2;
        *(_WORD *)v13 = 29500;
        *(_QWORD *)(a2 + 32) += 2LL;
        goto LABEL_10;
      }
LABEL_29:
      v9 = sub_CB6200(a2, "<s", 2u);
LABEL_10:
      v10 = sub_CB59D0(v9, v8);
      v11 = *(_BYTE **)(v10 + 32);
      if ( *(_BYTE **)(v10 + 24) == v11 )
      {
        v10 = sub_CB6200(v10, (unsigned __int8 *)">", 1u);
      }
      else
      {
        *v11 = 62;
        ++*(_QWORD *)(v10 + 32);
      }
      sub_C67200((__int64 *)v34, (__int64)&v31);
      sub_CB6200(v10, v34[0], (size_t)v34[1]);
      if ( (__int64 *)v34[0] != &v35 )
        j_j___libc_free_0(v34[0], v35 + 1);
      goto LABEL_14;
    }
    if ( v31 == (unsigned __int8 *)v33 )
      goto LABEL_17;
    j_j___libc_free_0(v31, v33[0] + 1LL);
    if ( v30 == v8 )
      return v7;
LABEL_18:
    ++v8;
  }
  while ( v8 != 64 );
  if ( (_BYTE)v7 )
  {
    v23 = *(__m128i **)(a2 + 32);
    v24 = *(unsigned __int8 *)(a1 + 16);
    v25 = *(_QWORD *)(a2 + 24) - (_QWORD)v23;
    if ( (_BYTE)v24 )
    {
      if ( v25 <= 0x2B )
      {
        sub_CB6200(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>", 0x2Cu);
      }
      else
      {
        v26 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
        qmemcpy(&v23[2], "ated...</td>", 12);
        *v23 = v26;
        v23[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB40);
        *(_QWORD *)(a2 + 32) += 44LL;
      }
      return v24;
    }
    else if ( v25 <= 0x11 )
    {
      sub_CB6200(a2, "|<s64>truncated...", 0x12u);
    }
    else
    {
      v28 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
      v23[1].m128i_i16[0] = 11822;
      *v23 = v28;
      *(_QWORD *)(a2 + 32) += 18LL;
    }
  }
  return v7;
}
