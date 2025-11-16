// Function: sub_2E8D910
// Address: 0x2e8d910
//
__int64 __fastcall sub_2E8D910(__int64 a1)
{
  __int64 v1; // rcx
  __int64 i; // rax
  __int64 v3; // r8
  unsigned __int8 v4; // dl
  __int64 *v5; // rdx
  __int64 v6; // rdx

  if ( (*(_DWORD *)(a1 + 40) & 0xFFFFFF) != 0 )
  {
    v1 = *(_QWORD *)(a1 + 32);
    for ( i = v1 + 40LL * ((*(_DWORD *)(a1 + 40) & 0xFFFFFFu) - 1); ; i -= 40 )
    {
      if ( *(_BYTE *)i == 14 )
      {
        v3 = *(_QWORD *)(i + 24);
        if ( v3 )
        {
          v4 = *(_BYTE *)(v3 - 16);
          if ( (v4 & 2) != 0 )
          {
            if ( !*(_DWORD *)(v3 - 24) )
              goto LABEL_3;
            v5 = *(__int64 **)(v3 - 32);
          }
          else
          {
            if ( (*(_WORD *)(v3 - 16) & 0x3C0) == 0 )
              goto LABEL_3;
            v5 = (__int64 *)(v3 + -16 - 8LL * ((v4 >> 2) & 0xF));
          }
          v6 = *v5;
          if ( *(_BYTE *)v6 == 1 && **(_BYTE **)(v6 + 136) == 17 )
            return *(_QWORD *)(i + 24);
        }
      }
LABEL_3:
      if ( i == v1 )
        return 0;
    }
  }
  return 0;
}
