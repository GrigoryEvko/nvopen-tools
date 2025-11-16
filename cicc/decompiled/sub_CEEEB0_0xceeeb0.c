// Function: sub_CEEEB0
// Address: 0xceeeb0
//
__int64 __fastcall sub_CEEEB0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  unsigned int v3; // eax
  unsigned int v4; // r15d
  const char *v5; // rax
  __int64 v6; // rdx

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 == a1 + 24 )
  {
    return 0;
  }
  else
  {
    while ( 1 )
    {
      v2 = v1 - 56;
      if ( !v1 )
        v2 = 0;
      LOBYTE(v3) = sub_B2FC80(v2);
      v4 = v3;
      if ( (_BYTE)v3 )
      {
        if ( *(_QWORD *)(v2 + 16) )
        {
          if ( (*(_BYTE *)(v2 + 33) & 0x20) == 0 )
          {
            v5 = sub_BD5D20(v2);
            if ( v6 != 14
              || *(_QWORD *)v5 != 0x725F6D76766E5F5FLL
              || *((_DWORD *)v5 + 2) != 1701602917
              || *((_WORD *)v5 + 6) != 29795 )
            {
              break;
            }
          }
        }
      }
      v1 = *(_QWORD *)(v1 + 8);
      if ( a1 + 24 == v1 )
        return 0;
    }
  }
  return v4;
}
