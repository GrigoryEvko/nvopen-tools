// Function: sub_15494B0
// Address: 0x15494b0
//
__int64 __fastcall sub_15494B0(_QWORD *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  _WORD *v9; // rdx

  if ( !*((_DWORD *)a1 + 120) )
    sub_16032E0(a2, a1 + 59);
  v5 = *a1;
  v6 = *(void **)(*a1 + 24LL);
  if ( *(_QWORD *)(*a1 + 16LL) - (_QWORD)v6 <= 0xBu )
  {
    sub_16E7EE0(v5, " syncscope(\"", 12);
  }
  else
  {
    qmemcpy(v6, " syncscope(\"", 12);
    *(_QWORD *)(v5 + 24) += 12LL;
  }
  v7 = (_QWORD *)(a1[59] + 16LL * a3);
  sub_16D16F0(*v7, v7[1], *a1);
  v8 = *a1;
  v9 = *(_WORD **)(*a1 + 24LL);
  if ( *(_QWORD *)(*a1 + 16LL) - (_QWORD)v9 <= 1u )
    return sub_16E7EE0(v8, "\")", 2);
  *v9 = 10530;
  *(_QWORD *)(v8 + 24) += 2LL;
  return 10530;
}
