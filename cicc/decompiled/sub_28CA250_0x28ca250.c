// Function: sub_28CA250
// Address: 0x28ca250
//
__int64 (__fastcall *__fastcall sub_28CA250(__int64 a1, __int64 a2, char a3))(_QWORD *, _QWORD *, __int64)
{
  _BYTE *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  void *v6; // rdx

  v3 = (_BYTE *)a2;
  if ( a3 )
    sub_904010(a2, "ExpressionTypeConstant, ");
  v4 = sub_904010(a2, "opcode = ");
  v5 = sub_CB59D0(v4, *(unsigned int *)(a1 + 12));
  sub_904010(v5, ", ");
  v6 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 0xBu )
  {
    v3 = (_BYTE *)sub_CB6200(a2, " constant = ", 0xCu);
  }
  else
  {
    qmemcpy(v6, " constant = ", 12);
    *(_QWORD *)(a2 + 32) += 12LL;
  }
  return sub_A69870(*(_QWORD *)(a1 + 24), v3, 0);
}
