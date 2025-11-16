// Function: sub_161E980
// Address: 0x161e980
//
__int64 __fastcall sub_161E980(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 v3; // r13

  v2 = 8LL * a2;
  v3 = sub_22077B0(v2 + a1) + v2;
  if ( v3 != v3 - v2 )
    memset((void *)(v3 - v2), 0, 8LL * a2);
  return v3;
}
