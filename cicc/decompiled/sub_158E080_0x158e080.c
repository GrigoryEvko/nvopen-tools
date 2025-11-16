// Function: sub_158E080
// Address: 0x158e080
//
__int64 __fastcall sub_158E080(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // eax
  unsigned int v4; // eax
  unsigned int v6; // eax

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 > a3 )
  {
    sub_158D430(a1, a2, a3);
    return a1;
  }
  else if ( v3 < a3 )
  {
    sub_158D100(a1, a2, a3);
    return a1;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
    {
      sub_16A4FD0(a1, a2);
      v6 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v6;
      if ( v6 <= 0x40 )
        goto LABEL_5;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
      v4 = *(_DWORD *)(a2 + 24);
      *(_DWORD *)(a1 + 24) = v4;
      if ( v4 <= 0x40 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
        return a1;
      }
    }
    sub_16A4FD0(a1 + 16, a2 + 16);
    return a1;
  }
}
