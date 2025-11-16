// Function: sub_1E5EBB0
// Address: 0x1e5ebb0
//
bool __fastcall sub_1E5EBB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax

  if ( a2 == a3 )
    return 1;
  v3 = sub_1E5E8E0(a1, a3);
  v4 = sub_1E5E8E0(a1, a2);
  return sub_1E5EAE0(a1, v4, v3);
}
