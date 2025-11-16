// Function: sub_1C31B60
// Address: 0x1c31b60
//
__int64 *__fastcall sub_1C31B60(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r12
  void *v5; // rdx
  const char *v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  char *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // rdx
  size_t v15; // [rsp+8h] [rbp-28h]

  if ( a3 == 2 && !byte_4FBA540 )
    return sub_16E8D30();
  sub_1C31A90(a3, *(_QWORD *)(a1 + 24));
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(void **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0xBu )
  {
    v4 = sub_16E7EE0(*(_QWORD *)(a1 + 24), ": Function `", 0xCu);
  }
  else
  {
    qmemcpy(v5, ": Function `", 12);
    *(_QWORD *)(v4 + 24) += 12LL;
  }
  v6 = sub_1649960(a2);
  v8 = *(_BYTE **)(v4 + 24);
  v9 = (char *)v6;
  v10 = *(_QWORD *)(v4 + 16) - (_QWORD)v8;
  if ( v10 < v7 )
  {
    v12 = sub_16E7EE0(v4, v9, v7);
    v8 = *(_BYTE **)(v12 + 24);
    v4 = v12;
    v10 = *(_QWORD *)(v12 + 16) - (_QWORD)v8;
  }
  else if ( v7 )
  {
    v15 = v7;
    memcpy(v8, v9, v7);
    v13 = *(_QWORD *)(v4 + 16);
    v14 = (_BYTE *)(*(_QWORD *)(v4 + 24) + v15);
    *(_QWORD *)(v4 + 24) = v14;
    v8 = v14;
    v10 = v13 - (_QWORD)v14;
  }
  if ( v10 <= 2 )
  {
    sub_16E7EE0(v4, "': ", 3u);
  }
  else
  {
    v8[2] = 32;
    *(_WORD *)v8 = 14887;
    *(_QWORD *)(v4 + 24) += 3LL;
  }
  return *(__int64 **)(a1 + 24);
}
