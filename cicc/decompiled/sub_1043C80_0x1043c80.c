// Function: sub_1043C80
// Address: 0x1043c80
//
__int64 __fastcall sub_1043C80(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  int v6; // ebx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // r14
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
  unsigned __int8 v23; // bl
  unsigned __int8 v25; // [rsp+Fh] [rbp-81h]
  __int64 v26; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v27; // [rsp+20h] [rbp-70h] BYREF
  size_t v28; // [rsp+28h] [rbp-68h]
  _QWORD v29[2]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int8 *v30[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h] BYREF

  v4 = (_QWORD *)(a3 + 48);
  v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v5 == v4 )
  {
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
    goto LABEL_43;
  }
  if ( !v5 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
LABEL_43:
    sub_904010(a2, "</tr><tr>");
    return 0;
  }
  v6 = sub_B46E30(v5 - 24);
  if ( *(_BYTE *)(a1 + 16) )
    sub_904010(a2, "</tr><tr>");
  if ( !v6 )
    return 0;
  v7 = (unsigned int)(v6 - 1);
  v25 = 0;
  v8 = 0;
  v26 = v7;
  do
  {
    sub_103B6D0((__int64)&v27, a3, v8);
    if ( v28 )
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
          *(_BYTE *)(v13 + 22) = 115;
          v15 = a2;
          *(_DWORD *)(v13 + 16) = 1953656688;
          *(_WORD *)(v13 + 20) = 8765;
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
        v19 = sub_CB6200(v18, v27, v28);
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
LABEL_15:
        if ( v27 != (unsigned __int8 *)v29 )
          j_j___libc_free_0(v27, v29[0] + 1LL);
        v25 = 1;
LABEL_18:
        if ( v26 == v8 )
          return v25;
        goto LABEL_19;
      }
      if ( (_DWORD)v8 )
      {
        if ( v12 != v13 )
        {
          *(_BYTE *)v13 = 124;
          v13 = *(_QWORD *)(a2 + 32) + 1LL;
          v21 = *(_QWORD *)(a2 + 24);
          *(_QWORD *)(a2 + 32) = v13;
          if ( (unsigned __int64)(v21 - v13) > 1 )
            goto LABEL_10;
          goto LABEL_30;
        }
        sub_CB6200(a2, (unsigned __int8 *)"|", 1u);
        v13 = *(_QWORD *)(a2 + 32);
        v12 = *(_QWORD *)(a2 + 24);
      }
      if ( (unsigned __int64)(v12 - v13) > 1 )
      {
LABEL_10:
        v9 = a2;
        *(_WORD *)v13 = 29500;
        *(_QWORD *)(a2 + 32) += 2LL;
        goto LABEL_11;
      }
LABEL_30:
      v9 = sub_CB6200(a2, "<s", 2u);
LABEL_11:
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
      sub_C67200((__int64 *)v30, (__int64)&v27);
      sub_CB6200(v10, v30[0], (size_t)v30[1]);
      if ( (__int64 *)v30[0] != &v31 )
        j_j___libc_free_0(v30[0], v31 + 1);
      goto LABEL_15;
    }
    if ( v27 == (unsigned __int8 *)v29 )
      goto LABEL_18;
    j_j___libc_free_0(v27, v29[0] + 1LL);
    if ( v26 == v8 )
      return v25;
LABEL_19:
    ++v8;
  }
  while ( v8 != 64 );
  if ( v25 )
  {
    v23 = *(_BYTE *)(a1 + 16);
    if ( v23 )
    {
      sub_904010(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>");
      return v23;
    }
    else
    {
      sub_904010(a2, "|<s64>truncated...");
    }
  }
  return v25;
}
