// Function: sub_1757250
// Address: 0x1757250
//
__int64 __fastcall sub_1757250(int *a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  unsigned int v4; // r14d
  bool v5; // al

  LOBYTE(v2) = sub_15FF7F0(*a1);
  v3 = v2;
  if ( !(_BYTE)v2 )
    return v3;
  v4 = *(_DWORD *)(a2 + 8);
  if ( v4 <= 0x40 )
    v5 = *(_QWORD *)a2 == 0;
  else
    v5 = v4 == (unsigned int)sub_16A57B0(a2);
  if ( v5 )
  {
    LOBYTE(v3) = (unsigned int)(*a1 - 32) > 1;
    return v3;
  }
  if ( v4 <= 0x40 )
  {
    if ( *(_QWORD *)a2 != 1 )
    {
      LOBYTE(v3) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *(_QWORD *)a2;
      goto LABEL_11;
    }
LABEL_14:
    if ( *a1 == 40 )
    {
      *a1 = 41;
      return v3;
    }
    return 0;
  }
  if ( (unsigned int)sub_16A57B0(a2) == v4 - 1 )
    goto LABEL_14;
  LOBYTE(v3) = v4 == (unsigned int)sub_16A58F0(a2);
LABEL_11:
  if ( !(_BYTE)v3 )
    return v3;
  if ( *a1 == 38 )
  {
    *a1 = 39;
    return v3;
  }
  return 0;
}
