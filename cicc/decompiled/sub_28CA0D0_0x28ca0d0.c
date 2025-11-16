// Function: sub_28CA0D0
// Address: 0x28ca0d0
//
__int64 (__fastcall *__fastcall sub_28CA0D0(__int64 a1, __int64 a2, char a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  _BYTE *v7; // rax

  if ( a3 )
    sub_904010(a2, "ExpressionTypeVariable, ");
  v4 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) <= 8 )
  {
    v5 = sub_CB6200(a2, "opcode = ", 9u);
  }
  else
  {
    *(_BYTE *)(v4 + 8) = 32;
    v5 = a2;
    *(_QWORD *)v4 = 0x3D2065646F63706FLL;
    *(_QWORD *)(a2 + 32) += 9LL;
  }
  v6 = sub_CB59D0(v5, *(unsigned int *)(a1 + 12));
  sub_904010(v6, ", ");
  v7 = (_BYTE *)sub_904010(a2, " variable = ");
  return sub_A69870(*(_QWORD *)(a1 + 24), v7, 0);
}
