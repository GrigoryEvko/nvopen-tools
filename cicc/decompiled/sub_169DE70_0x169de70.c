// Function: sub_169DE70
// Address: 0x169de70
//
__int64 __fastcall sub_169DE70(__int64 a1)
{
  _DWORD *v1; // r12
  __int64 v2; // rbx
  unsigned int v3; // r8d

  v1 = *(_DWORD **)(a1 + 8);
  v2 = a1;
  if ( v1 == sub_16982C0() )
    v2 = *(_QWORD *)(a1 + 16);
  v3 = 0;
  if ( (*(_BYTE *)(v2 + 26) & 7) == 1 )
    LOBYTE(v3) = v1[1] <= 0x3Fu;
  return v3;
}
