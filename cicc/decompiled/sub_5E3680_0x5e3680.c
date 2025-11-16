// Function: sub_5E3680
// Address: 0x5e3680
//
void __fastcall sub_5E3680(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbx
  int v3; // r14d
  __int64 v4; // rax
  __int64 i; // rbx

  v2 = *(_QWORD *)(a1 + 144);
  if ( !v2 )
    return;
  v3 = 0;
  do
  {
    while ( (*(_DWORD *)(v2 + 192) & 0x8000400) != 0 )
    {
LABEL_8:
      v2 = *(_QWORD *)(v2 + 112);
      if ( !v2 )
        goto LABEL_12;
    }
    if ( (*(_BYTE *)(v2 + 202) & 0x40) == 0 )
    {
      v4 = *(_QWORD *)(v2 + 256);
      if ( v4 && *(_QWORD *)(v4 + 24) && (*(_BYTE *)(v2 + 203) & 0x20) == 0 )
      {
        if ( a2 )
          sub_5E13C0(v2, (FILE *)(*(_DWORD *)(v2 + 160) != 0));
      }
      else
      {
        sub_5E13C0(v2, (FILE *)a2);
      }
      goto LABEL_8;
    }
    v2 = *(_QWORD *)(v2 + 112);
    v3 = 1;
  }
  while ( v2 );
LABEL_12:
  if ( (v3 & (a2 ^ 1)) != 0 )
  {
    for ( i = *(_QWORD *)(a1 + 144); i; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_BYTE *)(i + 202) & 0x40) != 0 )
        sub_5E13C0(i, 0);
    }
  }
}
