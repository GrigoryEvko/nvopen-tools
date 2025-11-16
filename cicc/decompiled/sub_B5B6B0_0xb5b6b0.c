// Function: sub_B5B6B0
// Address: 0xb5b6b0
//
__int64 __fastcall sub_B5B6B0(__int64 a1)
{
  unsigned __int8 *v1; // r8
  int v2; // edx
  __int64 v4; // rax
  unsigned __int64 v5; // r8
  int v6; // eax
  unsigned __int64 v7; // r8

  v1 = *(unsigned __int8 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v2 = *v1;
  if ( (unsigned int)(v2 - 12) <= 1 )
    return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( (_BYTE)v2 == 21 )
    return sub_ACA8A0(*((__int64 ***)v1 + 1));
  if ( (_BYTE)v2 != 95 )
    return *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v4 = sub_AA5510(*((_QWORD *)v1 + 5));
  v5 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v4 + 48 )
    return 0;
  if ( !v5 )
    BUG();
  v6 = *(unsigned __int8 *)(v5 - 24);
  v7 = v5 - 24;
  if ( (unsigned int)(v6 - 30) >= 0xB )
    return 0;
  return v7;
}
