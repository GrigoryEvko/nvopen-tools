// Function: sub_1790AD0
// Address: 0x1790ad0
//
__int64 __fastcall sub_1790AD0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // ebx
  unsigned int v4; // r15d
  unsigned int v5; // ebx

  v3 = *(_DWORD *)(a1 + 8);
  if ( v3 <= 0x40 )
    LOBYTE(v2) = *(_QWORD *)a1 == 0;
  else
    LOBYTE(v2) = v3 == (unsigned int)sub_16A57B0(a1);
  if ( !(_BYTE)v2 )
  {
    v4 = *(_DWORD *)(a2 + 8);
    if ( v4 <= 0x40 )
    {
      if ( *(_QWORD *)a2 )
        return v2;
    }
    else if ( v4 != (unsigned int)sub_16A57B0(a2) )
    {
      return v2;
    }
  }
  if ( v3 <= 0x40 )
  {
    v2 = 1;
    if ( *(_QWORD *)a1 == 1 )
      return v2;
    LOBYTE(v2) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) == *(_QWORD *)a1;
  }
  else
  {
    v2 = 1;
    if ( (unsigned int)sub_16A57B0(a1) == v3 - 1 )
      return v2;
    LOBYTE(v2) = v3 == (unsigned int)sub_16A58F0(a1);
  }
  if ( !(_BYTE)v2 )
  {
    v5 = *(_DWORD *)(a2 + 8);
    if ( v5 <= 0x40 )
    {
      v2 = 1;
      if ( *(_QWORD *)a2 != 1 )
        LOBYTE(v2) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) == *(_QWORD *)a2;
    }
    else
    {
      v2 = 1;
      if ( (unsigned int)sub_16A57B0(a2) != v5 - 1 )
        LOBYTE(v2) = v5 == (unsigned int)sub_16A58F0(a2);
    }
  }
  return v2;
}
