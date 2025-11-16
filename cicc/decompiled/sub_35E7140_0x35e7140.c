// Function: sub_35E7140
// Address: 0x35e7140
//
__int64 __fastcall sub_35E7140(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rbx
  __int64 i; // r13

  v6 = 1;
  v7 = a2[6];
  for ( i = a2[7]; i != v7; v7 += 256 )
  {
    if ( (*(_BYTE *)(v7 + 254) & 1) == 0 )
      sub_2F8F5D0(v7, a2, a3, a4, a5, a6);
    if ( v6 < *(_DWORD *)(v7 + 240) + (unsigned int)*(unsigned __int16 *)(v7 + 252) )
      v6 = *(_DWORD *)(v7 + 240) + *(unsigned __int16 *)(v7 + 252);
  }
  return v6 * dword_5040608;
}
