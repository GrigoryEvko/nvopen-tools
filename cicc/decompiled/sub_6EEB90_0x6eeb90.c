// Function: sub_6EEB90
// Address: 0x6eeb90
//
__int64 __fastcall sub_6EEB90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  int v5; // r14d
  char v6; // al
  int v7; // r13d
  __int64 v9; // rcx
  __int64 v10; // r8

  v3 = a2;
  v4 = a1;
  if ( *(char *)(a1 + 142) >= 0 && *(_BYTE *)(a1 + 140) == 12 )
    v5 = sub_8D4AB0(a1, a2, a3);
  else
    v5 = *(_DWORD *)(a1 + 136);
  if ( *(char *)(a2 + 142) < 0 )
  {
    v7 = *(_DWORD *)(a2 + 136);
    goto LABEL_17;
  }
  v6 = *(_BYTE *)(a2 + 140);
  if ( v6 == 12 )
  {
    v7 = sub_8D4AB0(a2, a2, a3);
LABEL_17:
    v6 = *(_BYTE *)(a2 + 140);
    if ( *(_BYTE *)(a1 + 140) != 12 )
      goto LABEL_8;
    goto LABEL_7;
  }
  v7 = *(_DWORD *)(a2 + 136);
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    do
LABEL_7:
      v4 = *(_QWORD *)(v4 + 160);
    while ( *(_BYTE *)(v4 + 140) == 12 );
LABEL_8:
    if ( v6 == 12 )
    {
      do
        v3 = *(_QWORD *)(v3 + 160);
      while ( *(_BYTE *)(v3 + 140) == 12 );
    }
  }
  if ( !(unsigned int)sub_8D3350(v4) || !(unsigned int)sub_8D3350(v3) )
    return 0;
  if ( v3 == v4 || (unsigned int)sub_8D97D0(v3, v4, 0, v9, v10) )
    return 1;
  if ( (unsigned int)sub_8D2A90(v4) || (unsigned int)sub_8D2A90(v3) || (unsigned int)sub_8D29A0(v3) )
    return 0;
  return (*(_QWORD *)(v3 + 128) == *(_QWORD *)(v4 + 128)) & (unsigned __int8)(v5 == v7);
}
