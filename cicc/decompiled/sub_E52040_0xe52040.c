// Function: sub_E52040
// Address: 0xe52040
//
_BYTE *__fastcall sub_E52040(
        __int64 a1,
        char *a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        __int64 a8,
        char *a9,
        __int64 a10)
{
  char *v11; // r11
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v19; // [rsp+0h] [rbp-50h]

  v11 = a2;
  v14 = *(_QWORD *)(a1 + 304);
  v15 = *(_QWORD *)(v14 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v14 + 24) - v15) <= 6 )
  {
    v19 = a3;
    sub_CB6200(v14, "\t.file\t", 7u);
    v11 = a2;
    a3 = v19;
  }
  else
  {
    *(_DWORD *)v15 = 1768304137;
    *(_WORD *)(v15 + 4) = 25964;
    *(_BYTE *)(v15 + 6) = 9;
    *(_QWORD *)(v14 + 32) += 7LL;
  }
  sub_E51560(a1, v11, a3, *(_QWORD *)(a1 + 304));
  if ( !(a10 | a8 | a5) )
    return sub_E4D880(a1);
  v16 = *(_QWORD *)(a1 + 304);
  v17 = *(_BYTE **)(v16 + 32);
  if ( *(_BYTE **)(v16 + 24) == v17 )
  {
    sub_CB6200(v16, (unsigned __int8 *)",", 1u);
    if ( a8 )
    {
LABEL_6:
      sub_E51560(a1, a7, a8, *(_QWORD *)(a1 + 304));
      if ( !(a10 | a5) )
        return sub_E4D880(a1);
      goto LABEL_10;
    }
  }
  else
  {
    *v17 = 44;
    ++*(_QWORD *)(v16 + 32);
    if ( a8 )
      goto LABEL_6;
  }
  if ( !(a10 | a5) )
    return sub_E4D880(a1);
LABEL_10:
  sub_904010(*(_QWORD *)(a1 + 304), ",");
  if ( a5 )
    sub_E51560(a1, a4, a5, *(_QWORD *)(a1 + 304));
  if ( a10 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), ",");
    sub_E51560(a1, a9, a10, *(_QWORD *)(a1 + 304));
  }
  return sub_E4D880(a1);
}
