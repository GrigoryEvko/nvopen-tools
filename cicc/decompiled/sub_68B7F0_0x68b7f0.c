// Function: sub_68B7F0
// Address: 0x68b7f0
//
_BOOL8 __fastcall sub_68B7F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r12
  char v6; // dl
  __int64 v7; // rax
  char v8; // dl
  __int64 v9; // rax
  __int64 v11; // rax

  for ( i = *(_QWORD *)(a2 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_BYTE *)(a2 + 25) == 1 )
    sub_6FA3A0(a2 + 8);
  v6 = *(_BYTE *)(a1 + 140);
  if ( v6 == 12 )
  {
    v7 = a1;
    do
    {
      v7 = *(_QWORD *)(v7 + 160);
      v6 = *(_BYTE *)(v7 + 140);
    }
    while ( v6 == 12 );
  }
  if ( !v6 )
    return 1;
  v8 = *(_BYTE *)(i + 140);
  if ( v8 == 12 )
  {
    v9 = i;
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
      v8 = *(_BYTE *)(v9 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 )
    return 1;
  if ( dword_4F04C44 != -1
    || (v11 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v11 + 6) & 6) != 0)
    || *(_BYTE *)(v11 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DBE70(a1) || (unsigned int)sub_8DBE70(i) )
      return 1;
  }
  return a1 == i || (unsigned int)sub_8D97D0(a1, i, 0, a4, a5) != 0;
}
