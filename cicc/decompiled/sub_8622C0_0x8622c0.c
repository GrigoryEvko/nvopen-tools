// Function: sub_8622C0
// Address: 0x8622c0
//
void __fastcall sub_8622C0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 j; // rbx
  __int64 k; // rbx
  char v6; // al
  char v7; // al
  char v8; // al
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 m; // rbx
  __int64 v12; // rax
  bool v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax

  if ( !*(_BYTE *)(a1 + 28) )
    dword_4D03B60 = 1;
  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_8622C0(*(_QWORD *)(i + 128));
  }
  for ( j = *(_QWORD *)(a1 + 104); j; j = *(_QWORD *)(j + 112) )
  {
    if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u )
    {
      v9 = *(_QWORD *)(j + 168);
      a2 = 6;
      sub_7607C0(j, 6);
      v10 = *(_QWORD *)(v9 + 152);
      if ( v10 )
      {
        if ( (*(_BYTE *)(v10 + 29) & 0x20) == 0 )
          sub_8622C0(v10);
      }
    }
  }
  for ( k = *(_QWORD *)(a1 + 112); k; k = *(_QWORD *)(k + 112) )
  {
    if ( (*(_BYTE *)(k + 88) & 8) == 0 )
    {
      if ( (v6 = *(_BYTE *)(k + 156), (v6 & 1) == 0) || (v6 & 2) != 0 && !unk_4D045AC || (v6 & 0x40) != 0 )
      {
        if ( !(unsigned int)sub_8D2FF0(*(_QWORD *)(k + 120), a2) && !(unsigned int)sub_8D3030(*(_QWORD *)(k + 120)) )
        {
          v7 = *(_BYTE *)(k + 170);
          if ( (v7 & 0x10) != 0 )
          {
            if ( dword_4D03FE8[0] )
            {
              if ( v7 >= 0 )
              {
                v14 = sub_892240(*(_QWORD *)k, a2);
                if ( (*(_BYTE *)(v14 + 80) & 8) == 0 )
                {
                  v15 = *(_QWORD *)(v14 + 16);
                  if ( !v15 || (*(_BYTE *)(v15 + 28) & 6) != 2 )
                    goto LABEL_13;
                }
              }
              else if ( *(_BYTE *)(k + 136) == 1 )
              {
                goto LABEL_13;
              }
            }
          }
          else
          {
            v8 = *(_BYTE *)(k + 136);
            if ( !v8 && (*(_DWORD *)(k + 172) & 0x18000) == 0 || *(_BYTE *)(k + 177) == 2 )
            {
              if ( (*(_BYTE *)(k + 172) & 0x28) == 0x28 )
                goto LABEL_13;
              goto LABEL_12;
            }
            if ( !*(_QWORD *)(k + 224) )
            {
              if ( v8 == 2 )
              {
                if ( (*(_BYTE *)(k + 168) & 0x48) == 0 )
                  goto LABEL_50;
              }
              else if ( !*(_QWORD *)(k + 232) || v8 != 1 )
              {
LABEL_50:
                if ( (*(_BYTE *)(k + 169) & 8) != 0 )
                  goto LABEL_13;
              }
            }
          }
        }
      }
    }
LABEL_12:
    sub_7604D0(k, 7u);
LABEL_13:
    a2 = 7;
    sub_7607C0(k, 7);
  }
  for ( m = *(_QWORD *)(a1 + 144); m; m = *(_QWORD *)(m + 112) )
  {
    v12 = *(_QWORD *)(m + 256);
    v13 = (*(_BYTE *)(m + 193) & 0x20) != 0;
    if ( v12 && *(_QWORD *)(v12 + 8) && (*(_BYTE *)(m + 202) & 4) == 0 && *(_BYTE *)(m + 172) != 2 )
      sub_7604D0(m, 0xBu);
    *(_BYTE *)(m + 193) &= ~0x20u;
    sub_7607C0(m, 11);
    *(_BYTE *)(m + 193) = *(_BYTE *)(m + 193) & 0xDF | (32 * v13);
  }
  if ( !*(_BYTE *)(a1 + 28) )
  {
    if ( dword_4F077C4 == 2 )
      sub_85B370(a1);
    dword_4D03B60 = 0;
  }
}
