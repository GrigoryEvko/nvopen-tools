// Function: sub_80AFC0
// Address: 0x80afc0
//
__int64 __fastcall sub_80AFC0(__int64 a1, char a2)
{
  __int64 v2; // rax
  char v3; // al
  unsigned int v4; // r9d
  char v6; // r10
  __int64 v7; // rax
  __int64 v8; // rax

  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    goto LABEL_7;
  v2 = *(_QWORD *)(a1 + 40);
  if ( !v2 )
    goto LABEL_5;
  v3 = *(_BYTE *)(v2 + 28);
  if ( v3 == 3 )
  {
LABEL_7:
    v4 = 1;
    if ( a2 != 6 )
      return v4;
    if ( *(_BYTE *)(a1 + 140) != 9 )
      return v4;
    if ( !sub_80A5F0(a1) )
      return v4;
    if ( (v6 & 5) == 1 )
      return v4;
    v7 = *(_QWORD *)(a1 + 40);
    if ( v7 )
    {
      if ( *(_BYTE *)(v7 + 28) == 16 )
        return v4;
    }
    goto LABEL_13;
  }
  v4 = 1;
  if ( v3 == 16 )
    return v4;
LABEL_5:
  v4 = 0;
  if ( a2 != 6 || *(_BYTE *)(a1 + 140) != 9 )
    return v4;
LABEL_13:
  v8 = *(_QWORD *)(a1 + 168);
  v4 = 0;
  if ( (*(_BYTE *)(v8 + 109) & 0x20) != 0 )
    return (*(_BYTE *)(v8 + 110) & 0x18) != 0;
  return v4;
}
