// Function: sub_E55010
// Address: 0xe55010
//
_BYTE *__fastcall sub_E55010(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 *v8; // rsi
  __int64 *v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // rdi

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
  {
    v6 = *(_QWORD *)(a1 + 288);
    if ( v6 )
      sub_E7BC40(a1, *(_QWORD *)(v6 + 8));
  }
  sub_E547B0(a1, a2, a3);
  if ( *(_BYTE *)(a1 + 746) )
  {
    v7 = *(_QWORD *)(a1 + 320);
    v8 = (__int64 *)(a1 + 640);
    if ( !*(_BYTE *)(a1 + 745) )
      v8 = sub_CB7330();
    sub_E82600(a2, v8, v7, "\n ", 2, 0);
    v9 = (__int64 *)(a1 + 640);
    if ( !*(_BYTE *)(a1 + 745) )
      v9 = sub_CB7330();
    sub_904010((__int64)v9, "\n");
  }
  v10 = *(_QWORD *)(a1 + 16);
  if ( v10 )
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD, __int64, __int64, _QWORD))(*(_QWORD *)v10 + 32LL))(
      v10,
      *(_QWORD *)(a1 + 320),
      0,
      a2,
      a3,
      *(_QWORD *)(a1 + 304));
  else
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD, const char *, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 320)
                                                                                           + 32LL))(
      *(_QWORD *)(a1 + 320),
      a2,
      0,
      byte_3F871B3,
      0,
      a3,
      *(_QWORD *)(a1 + 304));
  v11 = *(_QWORD *)(a1 + 496);
  if ( v11 && *(_BYTE *)(*(_QWORD *)(a1 + 488) + v11 - 1) != 10 )
  {
    v12 = (__int64 *)(a1 + 640);
    if ( !*(_BYTE *)(a1 + 745) )
      v12 = sub_CB7330();
    sub_904010((__int64)v12, "\n");
  }
  return sub_E4D880(a1);
}
