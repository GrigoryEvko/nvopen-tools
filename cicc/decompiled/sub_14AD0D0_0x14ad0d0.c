// Function: sub_14AD0D0
// Address: 0x14ad0d0
//
bool __fastcall sub_14AD0D0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rax
  int v3; // edx
  bool result; // al

  v1 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  if ( (a1 & 4) != 0 )
    v1 = (__int64 *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v2 = *v1;
  if ( *(_BYTE *)(v2 + 16) )
    return 0;
  v3 = *(_DWORD *)(v2 + 36);
  result = 1;
  if ( v3 != 115 && v3 != 203 )
    return v3 == 3660;
  return result;
}
