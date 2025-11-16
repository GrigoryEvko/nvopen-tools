// Function: sub_15A59B0
// Address: 0x15a59b0
//
__int64 __fastcall sub_15A59B0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r10d
  __int64 v4; // r12

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 8);
  if ( a3 )
    v3 = sub_161FF10(*(_QWORD *)(a1 + 8), a2, a3);
  return sub_15BC830(v4, 59, v3, 0, 0, 0, 0, 1);
}
