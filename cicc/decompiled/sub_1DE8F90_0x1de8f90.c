// Function: sub_1DE8F90
// Address: 0x1de8f90
//
__int64 __fastcall sub_1DE8F90(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // ecx
  unsigned int v4; // r8d
  unsigned int v5; // eax
  __int64 v6; // rdi

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 15;
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 24);
    v2 = *(_QWORD *)(a1 + 16);
    if ( !v4 )
      return v4;
    v3 = v4 - 1;
  }
  v4 = 1;
  v5 = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = *(_QWORD *)(v2 + 8LL * v5);
  if ( v6 != a2 )
  {
    while ( 1 )
    {
      if ( v6 == -8 )
        return 0;
      v5 = v3 & (v4 + v5);
      v6 = *(_QWORD *)(v2 + 8LL * v5);
      if ( a2 == v6 )
        break;
      ++v4;
    }
    return 1;
  }
  return v4;
}
