// Function: sub_6E93E0
// Address: 0x6e93e0
//
__int64 __fastcall sub_6E93E0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rax
  unsigned int v7; // edi

  v1 = 0;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v3 = *(_QWORD *)a1;
    v4 = *(_BYTE *)(v3 + 140);
    if ( v4 == 12 )
    {
      v5 = v3;
      do
      {
        v5 = *(_QWORD *)(v5 + 160);
        v4 = *(_BYTE *)(v5 + 140);
      }
      while ( v4 == 12 );
    }
    v1 = 0;
    if ( v4 )
    {
      v1 = sub_8D2960(v3);
      if ( v1 )
      {
        return 1;
      }
      else
      {
        if ( unk_4D04778 )
        {
          v7 = 1304;
          if ( !unk_4D04000 )
          {
            v7 = 1303;
            if ( dword_4F077C4 == 2 )
            {
              v7 = 2141;
              if ( unk_4F07778 <= 201102 )
                v7 = dword_4F07774 == 0 ? 1303 : 2141;
            }
          }
        }
        else
        {
          v7 = sub_6E92F0();
        }
        sub_6E68E0(v7, a1);
      }
    }
  }
  return v1;
}
