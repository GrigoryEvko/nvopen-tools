// Function: sub_1E17820
// Address: 0x1e17820
//
__int64 __fastcall sub_1E17820(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rcx
  unsigned int v3; // edx
  _DWORD *v4; // rax
  __int64 v5; // rdx

  if ( !**(_WORD **)(a1 + 16) || (v1 = 0, **(_WORD **)(a1 + 16) == 45) )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_DWORD *)(a1 + 40);
    v1 = *(_DWORD *)(v2 + 48);
    if ( v3 > 3 )
    {
      v4 = (_DWORD *)(v2 + 128);
      v5 = v2 + 80LL * ((v3 - 4) >> 1) + 208;
      while ( v1 == *v4 )
      {
        v4 += 20;
        if ( v4 == (_DWORD *)v5 )
          return v1;
      }
      return 0;
    }
  }
  return v1;
}
