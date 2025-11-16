// Function: sub_19FE020
// Address: 0x19fe020
//
__int64 __fastcall sub_19FE020(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  char v6; // al
  __int64 v8; // r12
  int v9; // esi

  v5 = *a1;
  v6 = *(_BYTE *)(*a1 + 8);
  if ( v6 == 16 )
    v6 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
  if ( v6 == 11 )
    return sub_15FB530(a1, a2, a3, v5);
  v8 = sub_15FB5B0(a1, a2, a3, v5);
  v9 = *(_BYTE *)(a4 + 17) >> 1;
  if ( v9 == 127 )
    v9 = -1;
  sub_15F2440(v8, v9);
  return v8;
}
