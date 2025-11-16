// Function: sub_8E34C0
// Address: 0x8e34c0
//
__int64 __fastcall sub_8E34C0(__int64 a1)
{
  unsigned int v1; // r8d
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // rdx

  v1 = unk_4F04C28;
  if ( unk_4F04C28 )
  {
    while ( 1 )
    {
      v3 = *(_BYTE *)(a1 + 140);
      if ( v3 != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    v1 = 0;
    if ( (*(_BYTE *)(a1 + 141) & 0x20) != 0 && (unsigned __int8)(v3 - 9) <= 2u )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL);
      if ( v4 )
      {
        if ( (*(_BYTE *)(v4 + 29) & 0x20) == 0 )
        {
          v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          v6 = *(_BYTE *)(v5 + 4);
          if ( v6 == 3 || !v6 )
            return 0;
          while ( v6 != 6 )
          {
            if ( v6 != 9 )
            {
              v5 = qword_4F04C68[0] + 776LL * *(int *)(v5 + 552);
              v6 = *(_BYTE *)(v5 + 4);
              if ( v6 )
              {
                if ( v6 != 3 )
                  continue;
              }
            }
            return 0;
          }
          v7 = *(_QWORD *)(v5 + 208);
          if ( v7 )
          {
            while ( v7 != a1 )
            {
              if ( dword_4F07588 )
              {
                v8 = *(_QWORD *)(v7 + 32);
                if ( *(_QWORD *)(a1 + 32) == v8 )
                {
                  if ( v8 )
                    break;
                }
              }
              if ( (*(_BYTE *)(v7 + 89) & 4) == 0 )
                return 0;
              v7 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 32LL);
              if ( !v7 )
                return 0;
            }
            return 1;
          }
          else
          {
            return 0;
          }
        }
      }
    }
  }
  return v1;
}
