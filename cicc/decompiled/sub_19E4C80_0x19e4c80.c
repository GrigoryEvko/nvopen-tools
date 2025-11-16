// Function: sub_19E4C80
// Address: 0x19e4c80
//
_BYTE *__fastcall sub_19E4C80(__int64 *a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  __int64 v4; // rax

  if ( a3 )
    sub_1263B40(a2, "ExpressionTypeStore, ");
  sub_1930810((__int64)a1, a2, 0);
  v3 = sub_1263B40(a2, " represents Store  ");
  sub_155C2B0(a1[7], v3, 0);
  sub_1263B40(a2, " with StoredValue ");
  sub_15537D0(a1[8], a2, 1, 0);
  v4 = sub_1263B40(a2, " and MemoryLeader ");
  return sub_14236E0(a1[6], v4);
}
