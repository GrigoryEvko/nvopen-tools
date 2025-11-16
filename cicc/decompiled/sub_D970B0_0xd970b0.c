// Function: sub_D970B0
// Address: 0xd970b0
//
__int64 __fastcall sub_D970B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned __int64 v5; // r14

  v3 = a2;
  v5 = sub_D97050(a1, a2);
  if ( v5 < sub_D97050(a1, a3) )
    return a3;
  return v3;
}
