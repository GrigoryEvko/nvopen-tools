// Function: sub_140CF00
// Address: 0x140cf00
//
__int64 __fastcall sub_140CF00(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  unsigned int v5; // eax

  v2 = *(_DWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
  {
    sub_16A4FD0(a1, a2 + 24);
    v5 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)(a2 + 24);
    v3 = *(_DWORD *)(a2 + 32);
    *(_DWORD *)(a1 + 24) = v3;
    if ( v3 <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 24);
      return a1;
    }
  }
  sub_16A4FD0(a1 + 16, a2 + 24);
  return a1;
}
