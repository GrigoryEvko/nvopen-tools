// Function: sub_15CC8F0
// Address: 0x15cc8f0
//
bool __fastcall sub_15CC8F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax

  if ( a2 == a3 )
    return 1;
  v3 = sub_15CC510(a1, a3);
  v4 = sub_15CC510(a1, a2);
  return sub_15CC7C0(a1, v4, v3);
}
