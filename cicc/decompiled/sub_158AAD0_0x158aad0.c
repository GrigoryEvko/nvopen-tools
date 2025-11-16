// Function: sub_158AAD0
// Address: 0x158aad0
//
__int64 __fastcall sub_158AAD0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned int v3; // eax
  unsigned int v5; // eax

  if ( !sub_158A0B0(a2) )
  {
    if ( !sub_158A670(a2) )
    {
LABEL_5:
      v3 = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 8) = v3;
      if ( v3 > 0x40 )
        sub_16A4FD0(a1, a2);
      else
        *(_QWORD *)a1 = *(_QWORD *)a2;
      return a1;
    }
    v2 = *(_DWORD *)(a2 + 24);
    if ( v2 <= 0x40 )
    {
      if ( !*(_QWORD *)(a2 + 16) )
        goto LABEL_5;
    }
    else if ( v2 == (unsigned int)sub_16A57B0(a2 + 16) )
    {
      goto LABEL_5;
    }
  }
  v5 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v5;
  if ( v5 <= 0x40 )
    *(_QWORD *)a1 = 0;
  else
    sub_16A4EF0(a1, 0, 0);
  return a1;
}
