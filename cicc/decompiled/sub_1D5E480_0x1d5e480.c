// Function: sub_1D5E480
// Address: 0x1d5e480
//
bool __fastcall sub_1D5E480(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // esi
  unsigned int v6; // eax
  __int64 v7; // r13
  unsigned __int8 v9; // al
  char v10; // al

  v4 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v4 <= 0x17u )
    return 0;
  v6 = sub_1F43D70(a1, (unsigned int)(v4 - 24));
  v7 = v6;
  if ( !v6 )
    return 1;
  v9 = sub_1D5D7E0(a2, *(__int64 **)a3, 0);
  if ( v9 != 1 && (!v9 || !*(_QWORD *)(a1 + 8LL * v9 + 120)) )
    return 0;
  if ( (unsigned int)v7 > 0x102 )
    return 1;
  v10 = *(_BYTE *)(v7 + 259LL * v9 + a1 + 2422);
  if ( !v10 )
    return 1;
  return v10 == 4;
}
