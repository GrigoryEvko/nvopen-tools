// Function: sub_130B9F0
// Address: 0x130b9f0
//
__int64 __fastcall sub_130B9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v8; // r13
  __int64 v9; // r8

  v8 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( !*(_QWORD *)(*(_QWORD *)(v8 + 8) + 56LL) )
    return 1;
  v9 = sub_13457A0(a1, a2, v8, a3, a5, a4 - a5, 0);
  if ( !v9 )
    return 1;
  sub_13453B0(a1, a2, v8, a2 + 56, v9);
  *a6 = 1;
  return 0;
}
