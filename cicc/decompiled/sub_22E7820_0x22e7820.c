// Function: sub_22E7820
// Address: 0x22e7820
//
__int64 __fastcall sub_22E7820(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v4; // rdx
  _QWORD *v5; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned int v8; // r15d
  unsigned int v9; // esi
  unsigned int v10; // ebx
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // r12
  _BYTE *v16; // rax
  unsigned __int64 v17; // r12
  unsigned int v18; // ebx
  unsigned __int8 v19; // bl
  __m128i *v20; // rdx
  unsigned __int64 v21; // rax
  __m128i v22; // xmm0
  __m128i si128; // xmm0
  __int64 v25; // rdi
  __int64 v26; // rax
  _WORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __m128i v32; // xmm0
  unsigned __int8 v34; // [rsp+13h] [rbp-9Dh]
  int v35; // [rsp+14h] [rbp-9Ch]
  __int64 v37; // [rsp+30h] [rbp-80h]
  unsigned __int64 v38; // [rsp+30h] [rbp-80h]
  _QWORD *v39; // [rsp+38h] [rbp-78h]
  _QWORD *v40; // [rsp+38h] [rbp-78h]
  unsigned __int8 *v41; // [rsp+40h] [rbp-70h] BYREF
  size_t v42; // [rsp+48h] [rbp-68h]
  _QWORD v43[2]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int8 *v44[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v45; // [rsp+70h] [rbp-40h] BYREF

  v39 = (_QWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 48);
  v4 = *v39 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = (_QWORD *)v4;
  if ( v39 == (_QWORD *)v4 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v4 )
      goto LABEL_12;
    v6 = 0;
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 < 0xB )
      v6 = v4 - 24;
    v7 = v6;
  }
  v8 = 0;
  v37 = v4 - 24;
  do
  {
    v10 = v8;
    if ( v39 != v5 )
    {
      if ( !v5 )
        goto LABEL_12;
      if ( (unsigned int)*((unsigned __int8 *)v5 - 24) - 30 <= 0xA )
      {
        if ( v8 == (unsigned int)sub_B46E30(v37) )
          break;
        goto LABEL_9;
      }
    }
    if ( !v8 )
      break;
LABEL_9:
    v9 = v8++;
  }
  while ( *(_QWORD *)(a3[1] + 32LL) == sub_B46EC0(v7, v9) );
  if ( v39 == v5 )
    goto LABEL_68;
  if ( !v5 )
LABEL_12:
    BUG();
  if ( (unsigned int)*((unsigned __int8 *)v5 - 24) - 30 <= 0xA )
  {
    v35 = sub_B46E30((__int64)(v5 - 3));
    goto LABEL_18;
  }
LABEL_68:
  v35 = 0;
LABEL_18:
  if ( *(_BYTE *)(a1 + 16) )
  {
    v31 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v31) <= 8 )
    {
      sub_CB6200(a2, "</tr><tr>", 9u);
    }
    else
    {
      *(_BYTE *)(v31 + 8) = 62;
      *(_QWORD *)v31 = 0x72743C3E72742F3CLL;
      *(_QWORD *)(a2 + 32) += 9LL;
    }
  }
  if ( v10 == v35 )
    return 0;
  v38 = 0;
  v11 = v10;
  v34 = 0;
  while ( 2 )
  {
    v41 = (unsigned __int8 *)v43;
    sub_22E4AB0((__int64 *)&v41, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v42 )
    {
      v12 = *(_QWORD *)(a2 + 24);
      v13 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(a1 + 16) )
      {
        if ( (unsigned __int64)(v12 - v13) <= 0x16 )
        {
          v25 = sub_CB6200(a2, "<td colspan=\"1\" port=\"s", 0x17u);
        }
        else
        {
          *(_BYTE *)(v13 + 22) = 115;
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
          *(_WORD *)(v13 + 20) = 8765;
          *(_DWORD *)(v13 + 16) = 1953656688;
          v25 = a2;
          *(__m128i *)v13 = si128;
          *(_QWORD *)(a2 + 32) += 23LL;
        }
        v26 = sub_CB59D0(v25, v38);
        v27 = *(_WORD **)(v26 + 32);
        v28 = v26;
        if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 1u )
        {
          v28 = sub_CB6200(v26, "\">", 2u);
        }
        else
        {
          *v27 = 15906;
          *(_QWORD *)(v26 + 32) += 2LL;
        }
        v29 = sub_CB6200(v28, v41, v42);
        v30 = *(_QWORD *)(v29 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v30) <= 4 )
        {
          sub_CB6200(v29, "</td>", 5u);
        }
        else
        {
          *(_DWORD *)v30 = 1685335868;
          *(_BYTE *)(v30 + 4) = 62;
          *(_QWORD *)(v29 + 32) += 5LL;
        }
      }
      else
      {
        if ( v38 )
        {
          if ( v12 == v13 )
          {
            sub_CB6200(a2, (unsigned __int8 *)"|", 1u);
            v13 = *(_QWORD *)(a2 + 32);
          }
          else
          {
            *(_BYTE *)v13 = 124;
            v13 = *(_QWORD *)(a2 + 32) + 1LL;
            *(_QWORD *)(a2 + 32) = v13;
          }
          v12 = *(_QWORD *)(a2 + 24);
        }
        if ( (unsigned __int64)(v12 - v13) <= 1 )
        {
          v14 = sub_CB6200(a2, "<s", 2u);
        }
        else
        {
          *(_WORD *)v13 = 29500;
          *(_QWORD *)(a2 + 32) += 2LL;
          v14 = a2;
        }
        v15 = sub_CB59D0(v14, v38);
        v16 = *(_BYTE **)(v15 + 32);
        if ( *(_BYTE **)(v15 + 24) == v16 )
        {
          v15 = sub_CB6200(v15, (unsigned __int8 *)">", 1u);
        }
        else
        {
          *v16 = 62;
          ++*(_QWORD *)(v15 + 32);
        }
        sub_C67200((__int64 *)v44, (__int64)&v41);
        sub_CB6200(v15, v44[0], (size_t)v44[1]);
        if ( (__int64 *)v44[0] != &v45 )
          j_j___libc_free_0((unsigned __int64)v44[0]);
      }
      if ( v41 != (unsigned __int8 *)v43 )
        j_j___libc_free_0((unsigned __int64)v41);
      v34 = 1;
    }
    else if ( v41 != (unsigned __int8 *)v43 )
    {
      j_j___libc_free_0((unsigned __int64)v41);
    }
    v40 = (_QWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v17 = *v40 & 0xFFFFFFFFFFFFFFF8LL;
    v18 = v11;
    while ( 2 )
    {
      ++v18;
      if ( (_QWORD *)v17 != v40 )
      {
        if ( !v17 )
          goto LABEL_12;
        if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 <= 0xA )
        {
          if ( v18 == (unsigned int)sub_B46E30(v17 - 24) )
            goto LABEL_41;
LABEL_36:
          if ( *(_QWORD *)(a3[1] + 32LL) != sub_B46EC0(v7, v18) )
            goto LABEL_41;
          continue;
        }
      }
      break;
    }
    if ( v18 )
      goto LABEL_36;
LABEL_41:
    v11 = v18;
    if ( v35 == v18 )
      return v34;
    if ( ++v38 != 64 )
      continue;
    break;
  }
  if ( v34 )
  {
    v19 = *(_BYTE *)(a1 + 16);
    v20 = *(__m128i **)(a2 + 32);
    v21 = *(_QWORD *)(a2 + 24) - (_QWORD)v20;
    if ( v19 )
    {
      if ( v21 <= 0x2B )
      {
        sub_CB6200(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>", 0x2Cu);
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
        qmemcpy(&v20[2], "ated...</td>", 12);
        *v20 = v22;
        v20[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB40);
        *(_QWORD *)(a2 + 32) += 44LL;
      }
      return v19;
    }
    else if ( v21 <= 0x11 )
    {
      sub_CB6200(a2, "|<s64>truncated...", 0x12u);
    }
    else
    {
      v32 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
      v20[1].m128i_i16[0] = 11822;
      *v20 = v32;
      *(_QWORD *)(a2 + 32) += 18LL;
    }
  }
  return v34;
}
