// Function: sub_6E9580
// Address: 0x6e9580
//
__int64 __fastcall sub_6E9580(__int64 a1)
{
  __int64 v2; // rdi
  char v3; // dl
  __int64 v4; // rax
  unsigned int v6; // eax

  if ( !*(_BYTE *)(a1 + 16) )
    return 0;
  v2 = *(_QWORD *)a1;
  v3 = *(_BYTE *)(v2 + 140);
  if ( v3 == 12 )
  {
    v4 = v2;
    do
    {
      v4 = *(_QWORD *)(v4 + 160);
      v3 = *(_BYTE *)(v4 + 140);
    }
    while ( v3 == 12 );
  }
  if ( !v3 )
    return 0;
  if ( !(unsigned int)sub_8D2D80(v2) )
  {
    v6 = sub_6E94D0();
    sub_6E68E0(v6, a1);
    return 0;
  }
  return 1;
}
