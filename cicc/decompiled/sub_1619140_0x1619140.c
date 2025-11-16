// Function: sub_1619140
// Address: 0x1619140
//
_QWORD *__fastcall sub_1619140(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r13
  const char *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  __int64 v12; // rax
  void *v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  size_t v16; // rdx
  _BYTE *v17; // rdi
  const char *v18; // rsi
  _BYTE *v19; // rax
  size_t v20; // r14
  __int64 v21; // rax
  __int64 v22; // r13
  const char *v23; // rax
  size_t v24; // rdx
  __int64 v25; // rax
  size_t v26; // [rsp+8h] [rbp-48h]
  void *v27; // [rsp+10h] [rbp-40h] BYREF
  const char *v28; // [rsp+18h] [rbp-38h]
  int v29; // [rsp+20h] [rbp-30h]

  if ( dword_4F9EA60 == -1 && qword_4F9E988 == qword_4F9E980 )
    return sub_16185C0(*(_QWORD *)(a1 + 16) + 568LL, a2);
  if ( (_BYTE)a3 )
  {
    v12 = sub_16E8CB0(a1, a2, a3);
    v13 = *(void **)(v12 + 24);
    v14 = v12;
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 <= 0xDu )
    {
      v14 = sub_16E7EE0(v12, "    DEFAULT   ", 14);
    }
    else
    {
      qmemcpy(v13, "    DEFAULT   ", 14);
      *(_QWORD *)(v12 + 24) += 14LL;
    }
    v15 = (*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
    v17 = *(_BYTE **)(v14 + 24);
    v18 = (const char *)v15;
    v19 = *(_BYTE **)(v14 + 16);
    v20 = v16;
    if ( v19 - v17 < v16 )
    {
      v14 = sub_16E7EE0(v14, v18, v16);
      v19 = *(_BYTE **)(v14 + 16);
      v17 = *(_BYTE **)(v14 + 24);
    }
    else if ( v16 )
    {
      memcpy(v17, v18, v16);
      v19 = *(_BYTE **)(v14 + 16);
      v17 = (_BYTE *)(v20 + *(_QWORD *)(v14 + 24));
      *(_QWORD *)(v14 + 24) = v17;
    }
    if ( v17 == v19 )
    {
      sub_16E7EE0(v14, "\n", 1);
    }
    else
    {
      *v17 = 10;
      ++*(_QWORD *)(v14 + 24);
    }
    return sub_16185C0(*(_QWORD *)(a1 + 16) + 568LL, a2);
  }
  ++dword_4F9E8C8;
  if ( (unsigned __int8)sub_160F9B0() )
  {
    v6 = sub_16E8CB0(a1, a2, v5);
    v28 = "%2d: ENABLED   ";
    v27 = &unk_49EDBF0;
    v29 = dword_4F9E8C8;
    v7 = sub_16E8450(v6, &v27);
    v8 = (const char *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
    v10 = *(void **)(v7 + 24);
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v10 < v9 )
    {
      v7 = sub_16E7EE0(v7, v8);
    }
    else if ( v9 )
    {
      v26 = v9;
      memcpy(v10, v8, v9);
      *(_QWORD *)(v7 + 24) += v26;
    }
    sub_1263B40(v7, "\n");
    return sub_16185C0(*(_QWORD *)(a1 + 16) + 568LL, a2);
  }
  v21 = sub_16E8CB0(a1, a2, v5);
  v28 = "%2d: DISABLED  ";
  v27 = &unk_49EDBF0;
  v29 = dword_4F9E8C8;
  v22 = sub_16E8450(v21, &v27);
  v23 = (const char *)(*(__int64 (__fastcall **)(__int64 *))(*a2 + 16))(a2);
  v25 = sub_1549FF0(v22, v23, v24);
  return (_QWORD *)sub_1263B40(v25, "\n");
}
