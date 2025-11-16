// Function: sub_E52660
// Address: 0xe52660
//
_BYTE *__fastcall sub_E52660(__int64 a1, __int64 a2, void *a3, size_t a4, char a5)
{
  __int64 v7; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // r13
  _WORD *v10; // rdx
  void *v11; // rdi
  __int64 v13; // rax
  void *src; // [rsp+0h] [rbp-30h] BYREF
  size_t n; // [rsp+8h] [rbp-28h]

  v7 = *(_QWORD *)(a1 + 304);
  src = a3;
  n = a4;
  v8 = *(_QWORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 7u )
  {
    sub_CB6200(v7, ".symver ", 8u);
  }
  else
  {
    *v8 = 0x207265766D79732ELL;
    *(_QWORD *)(v7 + 32) += 8LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    v13 = sub_CB6200(*(_QWORD *)(a1 + 304), (unsigned __int8 *)", ", 2u);
    v11 = *(void **)(v13 + 32);
    v9 = v13;
  }
  else
  {
    *v10 = 8236;
    v11 = (void *)(*(_QWORD *)(v9 + 32) + 2LL);
    *(_QWORD *)(v9 + 32) = v11;
  }
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v11 < n )
  {
    sub_CB6200(v9, (unsigned __int8 *)src, n);
LABEL_7:
    if ( a5 )
      return sub_E4D880(a1);
    goto LABEL_10;
  }
  if ( !n )
    goto LABEL_7;
  memcpy(v11, src, n);
  *(_QWORD *)(v9 + 32) += n;
  if ( a5 )
    return sub_E4D880(a1);
LABEL_10:
  if ( sub_C931B0((__int64 *)&src, word_3F645A0, 3u, 0) == -1 )
    sub_904010(*(_QWORD *)(a1 + 304), ", remove");
  return sub_E4D880(a1);
}
