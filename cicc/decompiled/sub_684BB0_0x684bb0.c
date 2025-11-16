// Function: sub_684BB0
// Address: 0x684bb0
//
__int64 __fastcall sub_684BB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned __int16 v4; // ax
  _BOOL4 *v5; // rsi
  __int64 p_key; // rdi
  unsigned int v7; // ebx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rbx
  _DWORD *v11; // rax
  bool v12; // cf
  bool v13; // zf
  unsigned int v15; // eax
  unsigned __int64 v16; // rax
  __int32 v18; // [rsp+14h] [rbp-6Ch]
  unsigned int v19; // [rsp+18h] [rbp-68h]
  __int16 v20; // [rsp+1Ch] [rbp-64h]
  __int8 v21; // [rsp+1Fh] [rbp-61h]
  _BOOL4 v22; // [rsp+2Ch] [rbp-54h] BYREF
  __m128i key; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+40h] [rbp-40h]

  v21 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL);
  sub_7C9660(a1);
  v18 = dword_4F063F8;
  v20 = word_4F063FC[0];
  v4 = word_4F06418[0];
  if ( word_4F06418[0] == 56 )
  {
    sub_7B8B50(a1, a2, v2, v3);
    v4 = word_4F06418[0];
  }
  v19 = 0;
  while ( 1 )
  {
    v22 = 0;
    if ( v4 == 4 )
    {
      v5 = &v22;
      v15 = sub_620FA0((__int64)xmmword_4F06300, &v22);
      v7 = v15;
      if ( v22 || (p_key = v15, v7 = sub_67D2F0(v15), v16 = v7 - 1LL, v22 = v16 > 0xED1) )
      {
        v5 = (_BOOL4 *)&dword_4F063F8;
        p_key = 1222;
        sub_684B30(0x4C6u, &dword_4F063F8);
      }
    }
    else if ( v4 == 1 )
    {
      p_key = (__int64)&key;
      v5 = (_BOOL4 *)&off_4A44400;
      v10 = *(_QWORD *)(qword_4D04A00 + 8);
      key.m128i_i64[0] = v10;
      v11 = bsearch(&key, &off_4A44400, 0xD2Cu, 0x10u, (__compar_fn_t)sub_67BDB0);
      v12 = 0;
      v13 = v11 == 0;
      if ( v11 )
      {
        v22 = 0;
        v7 = v11[2];
      }
      else
      {
        v22 = 1;
        v9 = 19;
        v5 = (_BOOL4 *)v10;
        p_key = (__int64)"vector_deprecation";
        do
        {
          if ( !v9 )
            break;
          v12 = *(_BYTE *)v5 < *(_BYTE *)p_key;
          v13 = *(_BYTE *)v5 == *(_BYTE *)p_key;
          v5 = (_BOOL4 *)((char *)v5 + 1);
          ++p_key;
          --v9;
        }
        while ( v13 );
        if ( (!v12 && !v13) == v12 )
        {
          v7 = 0;
          unk_4F5F770 = 1;
        }
        else
        {
          v5 = (_BOOL4 *)&dword_4F063F8;
          p_key = 1223;
          v7 = 0;
          sub_684B30(0x4C7u, &dword_4F063F8);
        }
      }
    }
    else
    {
      v5 = (_BOOL4 *)&dword_4F063F8;
      p_key = 1224;
      v22 = 1;
      v7 = 0;
      sub_684B30(0x4C8u, &dword_4F063F8);
    }
    sub_7B8B50(p_key, v5, v8, v9);
    if ( word_4F06418[0] != 9 && word_4F06418[0] != 67 )
    {
      sub_684B30(0xFDu, &dword_4F063F8);
      v19 = 1;
    }
    if ( !v22 )
      break;
LABEL_14:
    if ( !(unsigned int)sub_7BE800(67) )
      return sub_7C96B0(v19);
LABEL_15:
    v4 = word_4F06418[0];
  }
  byte_4CFFE80[4 * v7 + 2] |= 4u;
  if ( v21 == 34 )
  {
    sub_67D850(v7, 1, 0);
    goto LABEL_14;
  }
  if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 72) & 4) != 0 )
    {
      key.m128i_i8[9] &= ~1u;
      v24 = v7;
      key.m128i_i8[8] = v21;
      key.m128i_i32[0] = v18;
      key.m128i_i16[2] = v20;
      sub_67F350(&key);
    }
    goto LABEL_14;
  }
  key.m128i_i8[9] &= ~1u;
  v24 = v7;
  key.m128i_i8[8] = v21;
  key.m128i_i32[0] = v18;
  key.m128i_i16[2] = v20;
  sub_67F350(&key);
  if ( (unsigned int)sub_7BE800(67) )
    goto LABEL_15;
  return sub_7C96B0(v19);
}
