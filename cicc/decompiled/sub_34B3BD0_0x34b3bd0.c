// Function: sub_34B3BD0
// Address: 0x34b3bd0
//
_BOOL8 __fastcall sub_34B3BD0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  unsigned int v5; // esi
  unsigned int v6; // eax
  __int64 v7; // rax

  if ( *(_BYTE *)a3 )
    return 0;
  v3 = *(_BYTE *)(a3 + 3);
  if ( (v3 & 0x20) == 0 )
    return 0;
  v5 = *(_DWORD *)(a3 + 8);
  if ( v5 )
  {
    if ( (v3 & 0x10) != 0 )
    {
      v6 = sub_2E89C70(a2, v5, 0, 1);
      if ( v6 != -1 )
      {
LABEL_6:
        v7 = *(_QWORD *)(a2 + 32) + 40LL * v6;
        if ( v7 )
          return (*(_BYTE *)(v7 + 3) & 0x20) != 0;
      }
    }
    else
    {
      v6 = sub_2E8E710(a2, v5, 0, 0, 0);
      if ( v6 != -1 )
        goto LABEL_6;
    }
  }
  return 0;
}
