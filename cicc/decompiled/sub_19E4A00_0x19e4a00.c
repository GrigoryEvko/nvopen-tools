// Function: sub_19E4A00
// Address: 0x19e4a00
//
__int64 __fastcall sub_19E4A00(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( a3 )
    sub_1263B40(a2, "ExpressionTypeConstant, ");
  v3 = sub_1263B40(a2, "opcode = ");
  v4 = sub_16E7A90(v3, *(unsigned int *)(a1 + 12));
  sub_1263B40(v4, ", ");
  v5 = sub_1263B40(a2, " constant = ");
  return sub_155C2B0(*(_QWORD *)(a1 + 24), v5, 0);
}
