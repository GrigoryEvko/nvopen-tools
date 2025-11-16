// Function: sub_6E96B0
// Address: 0x6e96b0
//
__int64 __fastcall sub_6E96B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rax
  unsigned int v6; // edi

  result = 0;
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
    result = 0;
    if ( v4 )
    {
      if ( (unsigned int)sub_8D33B0() )
      {
        return 1;
      }
      else
      {
        v6 = 41;
        if ( !unk_4D04000 )
        {
          v6 = 849;
          if ( dword_4F077C4 == 2 )
          {
            v6 = 2139;
            if ( unk_4F07778 <= 201102 )
              v6 = dword_4F07774 == 0 ? 849 : 2139;
          }
        }
        sub_6E68E0(v6, a1);
        return 0;
      }
    }
  }
  return result;
}
