// Function: sub_19FE1F0
// Address: 0x19fe1f0
//
__int64 __fastcall sub_19FE1F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v6; // al
  __int64 v8; // r12
  int v9; // esi

  v6 = *(_BYTE *)(*a1 + 8);
  if ( v6 == 16 )
    v6 = *(_BYTE *)(**(_QWORD **)(*a1 + 16) + 8LL);
  if ( v6 == 11 )
    return sub_15FB440(15, a1, a2, a3, a4);
  v8 = sub_15FB440(16, a1, a2, a3, a4);
  v9 = *(_BYTE *)(a5 + 17) >> 1;
  if ( v9 == 127 )
    v9 = -1;
  sub_15F2440(v8, v9);
  return v8;
}
