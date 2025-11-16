// Function: sub_E59C30
// Address: 0xe59c30
//
_BYTE *__fastcall sub_E59C30(__int64 a1, unsigned __int8 a2, unsigned __int8 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx

  nullsub_349(a1, a2, a3);
  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xEu )
  {
    sub_CB6200(v5, "\t.cfi_sections ", 0xFu);
  }
  else
  {
    qmemcpy(v6, "\t.cfi_sections ", 15);
    *(_QWORD *)(v5 + 32) += 15LL;
  }
  if ( a2 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), ".eh_frame");
    if ( a3 )
      sub_904010(*(_QWORD *)(a1 + 304), ", .debug_frame");
  }
  else if ( a3 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), ".debug_frame");
  }
  return sub_E4D880(a1);
}
