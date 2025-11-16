// Function: sub_11F5CE0
// Address: 0x11f5ce0
//
__int64 __fastcall sub_11F5CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  int v6; // ebx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r13
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // r15
  _BYTE *v17; // rax
  __m128i si128; // xmm0
  __int64 v20; // rdi
  __int64 v21; // rax
  _WORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  __m128i *v27; // rdx
  unsigned __int8 v28; // bl
  unsigned __int64 v29; // rax
  __m128i v30; // xmm0
  __int64 v31; // rdx
  __m128i v32; // xmm0
  __int64 i; // [rsp+10h] [rbp-D0h]
  unsigned __int8 v35; // [rsp+1Fh] [rbp-C1h]
  __int64 v36; // [rsp+20h] [rbp-C0h]
  __int64 v37; // [rsp+28h] [rbp-B8h]
  __m128i *v38; // [rsp+30h] [rbp-B0h] BYREF
  size_t v39; // [rsp+38h] [rbp-A8h]
  __m128i v40; // [rsp+40h] [rbp-A0h] BYREF
  __m128i *v41; // [rsp+50h] [rbp-90h] BYREF
  size_t v42; // [rsp+58h] [rbp-88h]
  __m128i v43; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int8 *v44; // [rsp+70h] [rbp-70h] BYREF
  size_t v45; // [rsp+78h] [rbp-68h]
  _QWORD v46[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_QWORD *)(a3 + 48);
  v37 = a3 + 48;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3 + 48 == (v4 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
    v6 = 0;
    goto LABEL_60;
  }
  if ( !v5 )
LABEL_72:
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
    v6 = 0;
    if ( !*(_BYTE *)(a1 + 16) )
      return 0;
    goto LABEL_60;
  }
  v6 = sub_B46E30(v5 - 24);
  if ( *(_BYTE *)(a1 + 16) )
  {
LABEL_60:
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
    if ( v6 )
    {
      v4 = *(_QWORD *)(a3 + 48);
      goto LABEL_6;
    }
    return 0;
  }
  if ( !v6 )
    return 0;
LABEL_6:
  v7 = (unsigned int)(v6 - 1);
  v35 = 0;
  v8 = 0;
  v36 = v7;
  for ( i = a3; ; v4 = *(_QWORD *)(i + 48) )
  {
    v9 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v37 == v9 || !v9 || (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
      goto LABEL_72;
    v10 = *(_BYTE *)(v9 - 24);
    if ( v10 == 31 )
    {
      if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 3 )
        goto LABEL_8;
      v14 = *(_QWORD *)(a2 + 24);
      v38 = &v40;
      v11 = v14;
      v39 = 1;
      v13 = *(_BYTE *)(a1 + 16) == 0;
      v40.m128i_i16[0] = (unsigned __int8)((_DWORD)v8 == 0 ? 84 : 70);
      v12 = *(_QWORD *)(a2 + 32);
      if ( !v13 )
        goto LABEL_28;
      if ( !(_DWORD)v8 )
        goto LABEL_18;
    }
    else
    {
      if ( v10 != 32 )
        goto LABEL_8;
      if ( !(_DWORD)v8 )
      {
        v11 = *(_QWORD *)(a2 + 24);
        v12 = *(_QWORD *)(a2 + 32);
        v38 = &v40;
        v13 = *(_BYTE *)(a1 + 16) == 0;
        v14 = v11;
        v40.m128i_i32[0] = 6710628;
        v39 = 3;
        if ( !v13 )
          goto LABEL_28;
LABEL_18:
        if ( (unsigned __int64)(v11 - v12) > 1 )
          goto LABEL_19;
        goto LABEL_42;
      }
      v43.m128i_i8[0] = 0;
      v41 = &v43;
      v46[3] = 0x100000000LL;
      v46[4] = &v41;
      v45 = 0;
      v44 = (unsigned __int8 *)&unk_49DD210;
      v42 = 0;
      memset(v46, 0, 24);
      sub_CB5980((__int64)&v44, 0, 0, 0);
      sub_C49420(*(_QWORD *)(*(_QWORD *)(v9 - 32) + (v8 << 6)) + 24LL, (__int64)&v44, 1);
      v38 = &v40;
      if ( v41 == &v43 )
      {
        v40 = _mm_load_si128(&v43);
      }
      else
      {
        v38 = v41;
        v40.m128i_i64[0] = v43.m128i_i64[0];
      }
      v43.m128i_i8[0] = 0;
      v41 = &v43;
      v39 = v42;
      v42 = 0;
      v44 = (unsigned __int8 *)&unk_49DD210;
      sub_CB5840((__int64)&v44);
      if ( v41 != &v43 )
        j_j___libc_free_0(v41, v43.m128i_i64[0] + 1);
      if ( !v39 )
      {
        if ( v38 != &v40 )
          j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
LABEL_8:
        if ( v36 == v8 )
          return v35;
        goto LABEL_9;
      }
      v14 = *(_QWORD *)(a2 + 24);
      v12 = *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)(a1 + 16) )
      {
LABEL_28:
        if ( (unsigned __int64)(v14 - v12) <= 0x16 )
        {
          v20 = sub_CB6200(a2, "<td colspan=\"1\" port=\"s", 0x17u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
          *(_BYTE *)(v12 + 22) = 115;
          v20 = a2;
          *(_DWORD *)(v12 + 16) = 1953656688;
          *(_WORD *)(v12 + 20) = 8765;
          *(__m128i *)v12 = si128;
          *(_QWORD *)(a2 + 32) += 23LL;
        }
        v21 = sub_CB59D0(v20, v8);
        v22 = *(_WORD **)(v21 + 32);
        v23 = v21;
        if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 1u )
        {
          v23 = sub_CB6200(v21, "\">", 2u);
        }
        else
        {
          *v22 = 15906;
          *(_QWORD *)(v21 + 32) += 2LL;
        }
        v24 = sub_CB6200(v23, (unsigned __int8 *)v38, v39);
        v25 = *(_QWORD *)(v24 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) <= 4 )
        {
          sub_CB6200(v24, "</td>", 5u);
        }
        else
        {
          *(_DWORD *)v25 = 1685335868;
          *(_BYTE *)(v25 + 4) = 62;
          *(_QWORD *)(v24 + 32) += 5LL;
        }
        goto LABEL_24;
      }
    }
    if ( v12 == v14 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"|", 1u);
      v12 = *(_QWORD *)(a2 + 32);
      v11 = *(_QWORD *)(a2 + 24);
      goto LABEL_18;
    }
    *(_BYTE *)v12 = 124;
    v12 = *(_QWORD *)(a2 + 32) + 1LL;
    v26 = *(_QWORD *)(a2 + 24) - v12;
    *(_QWORD *)(a2 + 32) = v12;
    if ( v26 > 1 )
    {
LABEL_19:
      v15 = a2;
      *(_WORD *)v12 = 29500;
      *(_QWORD *)(a2 + 32) += 2LL;
      goto LABEL_20;
    }
LABEL_42:
    v15 = sub_CB6200(a2, "<s", 2u);
LABEL_20:
    v16 = sub_CB59D0(v15, v8);
    v17 = *(_BYTE **)(v16 + 32);
    if ( *(_BYTE **)(v16 + 24) == v17 )
    {
      v16 = sub_CB6200(v16, (unsigned __int8 *)">", 1u);
    }
    else
    {
      *v17 = 62;
      ++*(_QWORD *)(v16 + 32);
    }
    sub_C67200((__int64 *)&v44, (__int64)&v38);
    sub_CB6200(v16, v44, v45);
    if ( v44 != (unsigned __int8 *)v46 )
      j_j___libc_free_0(v44, v46[0] + 1LL);
LABEL_24:
    if ( v38 != &v40 )
      j_j___libc_free_0(v38, v40.m128i_i64[0] + 1);
    v35 = 1;
    if ( v36 == v8 )
      return v35;
LABEL_9:
    if ( ++v8 == 64 )
      break;
  }
  if ( v35 )
  {
    v27 = *(__m128i **)(a2 + 32);
    v28 = *(_BYTE *)(a1 + 16);
    v29 = *(_QWORD *)(a2 + 24) - (_QWORD)v27;
    if ( v28 )
    {
      if ( v29 <= 0x2B )
      {
        sub_CB6200(a2, "<td colspan=\"1\" port=\"s64\">truncated...</td>", 0x2Cu);
      }
      else
      {
        v30 = _mm_load_si128((const __m128i *)&xmmword_3F8CB30);
        qmemcpy(&v27[2], "ated...</td>", 12);
        *v27 = v30;
        v27[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB40);
        *(_QWORD *)(a2 + 32) += 44LL;
      }
      return v28;
    }
    else if ( v29 <= 0x11 )
    {
      sub_CB6200(a2, "|<s64>truncated...", 0x12u);
    }
    else
    {
      v32 = _mm_load_si128((const __m128i *)&xmmword_3F8CB50);
      v27[1].m128i_i16[0] = 11822;
      *v27 = v32;
      *(_QWORD *)(a2 + 32) += 18LL;
    }
  }
  return v35;
}
