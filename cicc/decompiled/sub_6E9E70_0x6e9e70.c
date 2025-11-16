// Function: sub_6E9E70
// Address: 0x6e9e70
//
__int64 __fastcall sub_6E9E70(__int64 *a1)
{
  char v1; // cl
  __int64 v4; // rdi
  char v5; // dl
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rax

  v1 = *((_BYTE *)a1 + 16);
  if ( !v1 )
    return 1;
  v4 = *a1;
  v5 = *(_BYTE *)(v4 + 140);
  if ( v5 == 12 )
  {
    v6 = v4;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v5 = *(_BYTE *)(v6 + 140);
    }
    while ( v5 == 12 );
  }
  if ( !v5 )
    return 1;
  if ( dword_4F04C44 == -1 )
  {
    v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v8 + 6) & 6) == 0 && *(_BYTE *)(v8 + 4) != 12 )
      return 0;
  }
  if ( v1 == 1 )
  {
    v7 = a1[18];
  }
  else
  {
    if ( !(unsigned int)sub_8DBE70(v4) )
      return 0;
    if ( *((_BYTE *)a1 + 16) != 2 || *((_BYTE *)a1 + 317) != 12 || *((_BYTE *)a1 + 320) != 1 )
      return 1;
    v7 = sub_72E9A0(a1 + 18);
  }
  if ( *(_BYTE *)(v7 + 24) )
    return sub_6DF920((__int64 *)v7);
  else
    return 1;
}
