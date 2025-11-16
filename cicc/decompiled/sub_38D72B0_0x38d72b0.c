// Function: sub_38D72B0
// Address: 0x38d72b0
//
__int64 __fastcall sub_38D72B0(__int64 a1, __int64 a2)
{
  int v3; // esi
  unsigned int v4; // eax
  __int64 v5; // rdi
  unsigned int v6; // edx
  int v7; // esi
  unsigned int v8; // r8d

  v3 = *(unsigned __int16 *)(a2 + 8);
  if ( v3 )
  {
    v4 = *(unsigned __int16 *)(a2 + 6);
    v5 = *(_QWORD *)(a1 + 144);
    v6 = 0;
    v7 = v4 + v3;
    while ( 1 )
    {
      v8 = *(__int16 *)(v5 + 4LL * v4);
      if ( *(__int16 *)(v5 + 4LL * v4) < 0 )
        break;
      if ( (int)v6 < (int)v8 )
        v6 = *(__int16 *)(v5 + 4LL * v4);
      if ( v7 == ++v4 )
        return v6;
    }
    return v8;
  }
  else
  {
    return 0;
  }
}
