// Function: sub_19E4B00
// Address: 0x19e4b00
//
__int64 __fastcall sub_19E4B00(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  if ( a3 )
    sub_1263B40(a2, "ExpressionTypeUnknown, ");
  v3 = sub_1263B40(a2, "opcode = ");
  v4 = sub_16E7A90(v3, *(unsigned int *)(a1 + 12));
  sub_1263B40(v4, ", ");
  v5 = sub_1263B40(a2, " inst = ");
  return sub_155C2B0(*(_QWORD *)(a1 + 24), v5, 0);
}
