// Function: sub_2B1F810
// Address: 0x2b1f810
//
__int64 __fastcall sub_2B1F810(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  unsigned int v6; // eax
  unsigned int v7; // eax
  int v8; // edx

  v4 = sub_DFDB60(a1);
  if ( !v4 )
    return 1;
  v5 = v4;
  if ( v4 >= a3 )
    return 1;
  if ( *(_BYTE *)(a2 + 8) != 17 )
    return 1;
  v6 = *(_DWORD *)(a2 + 32);
  if ( v5 >= v6 )
    return 1;
  v8 = v6 % v5;
  v7 = v6 / v5;
  if ( v8 || !sub_2B1F720(a1, *(_QWORD *)(a2 + 24), v7) )
    return 1;
  return v5;
}
