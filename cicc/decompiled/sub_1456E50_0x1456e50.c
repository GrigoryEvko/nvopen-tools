// Function: sub_1456E50
// Address: 0x1456e50
//
__int64 __fastcall sub_1456E50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned __int64 v5; // r14

  v3 = a2;
  v5 = sub_1456C90(a1, a2);
  if ( v5 < sub_1456C90(a1, a3) )
    return a3;
  return v3;
}
