// Function: sub_2E31DD0
// Address: 0x2e31dd0
//
__int64 __fastcall sub_2E31DD0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v7; // rax
  signed __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned int v10; // r9d

  v6 = *(_QWORD *)(a1 + 192);
  v7 = *(_QWORD *)(a1 + 184);
  v8 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v7) >> 3);
  if ( v8 >> 2 > 0 )
  {
    v9 = v7 + 96 * (v8 >> 2);
    while ( a2 != *(_DWORD *)v7 )
    {
      if ( a2 == *(_DWORD *)(v7 + 24) )
      {
        v7 += 24;
        goto LABEL_8;
      }
      if ( a2 == *(_DWORD *)(v7 + 48) )
      {
        v7 += 48;
        goto LABEL_8;
      }
      if ( a2 == *(_DWORD *)(v7 + 72) )
      {
        v7 += 72;
        goto LABEL_8;
      }
      v7 += 96;
      if ( v7 == v9 )
      {
        v8 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v7) >> 3);
        goto LABEL_12;
      }
    }
    goto LABEL_8;
  }
LABEL_12:
  if ( v8 == 2 )
  {
LABEL_18:
    if ( a2 == *(_DWORD *)v7 )
      goto LABEL_8;
    v7 += 24;
    goto LABEL_20;
  }
  if ( v8 == 3 )
  {
    if ( a2 == *(_DWORD *)v7 )
      goto LABEL_8;
    v7 += 24;
    goto LABEL_18;
  }
  if ( v8 != 1 )
    return 0;
LABEL_20:
  if ( a2 != *(_DWORD *)v7 )
    return 0;
LABEL_8:
  v10 = 0;
  if ( v6 != v7 )
    LOBYTE(v10) = (*(_QWORD *)(v7 + 16) & a4 | *(_QWORD *)(v7 + 8) & a3) != 0;
  return v10;
}
