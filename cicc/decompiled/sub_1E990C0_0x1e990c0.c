// Function: sub_1E990C0
// Address: 0x1e990c0
//
__int64 __fastcall sub_1E990C0(__int64 a1, __int64 a2)
{
  int v2; // edx

  if ( dword_4FC83C0 == -1 )
    return sub_1E968B0(a2);
  v2 = dword_4FC8218++;
  if ( v2 == dword_4FC83C0 )
    return sub_1E968B0(a2);
  else
    return 0;
}
