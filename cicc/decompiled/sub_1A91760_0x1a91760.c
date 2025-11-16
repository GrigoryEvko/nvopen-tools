// Function: sub_1A91760
// Address: 0x1a91760
//
__int64 __fastcall sub_1A91760(_QWORD *a1)
{
  unsigned __int64 v2; // rax
  unsigned int v3; // r8d
  unsigned __int8 v4; // dl
  __int64 v5; // rdi
  __int64 v7; // rdi
  unsigned __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // dl
  int v13; // eax

  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*a1 & 4) != 0 )
  {
    v3 = 0;
    if ( *(_BYTE *)(*(_QWORD *)(v2 - 24) + 16LL) == 20 )
      return v3;
  }
  v4 = *(_BYTE *)(v2 + 16);
  v5 = 0;
  if ( v4 <= 0x17u )
    goto LABEL_7;
  if ( v4 != 78 )
  {
    if ( v4 == 29 )
      v5 = v2;
LABEL_7:
    if ( sub_1642CF0(v5) )
      return 0;
    goto LABEL_11;
  }
  if ( sub_1642CF0(v2 | 4) )
    return 0;
LABEL_11:
  v7 = 0;
  v8 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 > 0x17u )
  {
    if ( v9 == 78 )
    {
      v7 = v8 | 4;
    }
    else if ( v9 == 29 )
    {
      v7 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  if ( sub_1642D80(v7) )
    return 0;
  v10 = 0;
  v11 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = *(_BYTE *)(v11 + 16);
  if ( v12 > 0x17u )
  {
    if ( v12 == 78 )
    {
      v10 = v11 | 4;
    }
    else if ( v12 == 29 )
    {
      v10 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  LOBYTE(v13) = sub_1642DF0(v10);
  return v13 ^ 1u;
}
