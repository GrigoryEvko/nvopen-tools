// Function: sub_19E4990
// Address: 0x19e4990
//
__int64 __fastcall sub_19E4990(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  if ( a3 )
  {
    v6 = sub_1263B40(a2, "etype = ");
    v7 = sub_16E7AB0(v6, *(int *)(a1 + 8));
    sub_1263B40(v7, ",");
  }
  v3 = sub_1263B40(a2, "opcode = ");
  v4 = sub_16E7A90(v3, *(unsigned int *)(a1 + 12));
  return sub_1263B40(v4, ", ");
}
