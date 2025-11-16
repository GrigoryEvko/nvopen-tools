// Function: sub_1CCB380
// Address: 0x1ccb380
//
__int64 __fastcall sub_1CCB380(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v2 = sub_1625940(a1, "nvvm.as", 7u);
  v3 = 0;
  if ( v2 )
  {
    if ( *(_DWORD *)(v2 + 8) == 1 )
    {
      v4 = *(_QWORD *)(v2 - 8);
      if ( *(_BYTE *)v4 == 1 )
      {
        v5 = *(_QWORD *)(v4 + 136);
        if ( *(_BYTE *)(v5 + 16) == 13 )
        {
          if ( *(_DWORD *)(v5 + 32) <= 0x40u )
            v6 = *(_QWORD *)(v5 + 24);
          else
            v6 = **(_QWORD **)(v5 + 24);
          *a2 = v6;
          return 1;
        }
      }
    }
  }
  return v3;
}
