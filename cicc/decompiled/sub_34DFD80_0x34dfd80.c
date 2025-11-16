// Function: sub_34DFD80
// Address: 0x34dfd80
//
_BOOL8 __fastcall sub_34DFD80(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r8
  __int64 v6; // r12
  int v8; // ecx
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // di
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 i; // rsi
  _BOOL4 v15; // edx
  __int64 v17; // rax
  int v18; // [rsp-2Ch] [rbp-2Ch]

  if ( a3 == a2 )
    return 0;
  v4 = a2;
  v6 = a4 >> 5;
  v8 = a4 & 0x1F;
  v9 = 4 * v6;
  do
  {
    v10 = *(_QWORD *)(v4 + 40);
    v11 = *(_BYTE *)(v10 + 3) & 0x10;
    if ( v11 && (*(_BYTE *)(v10 + 4) & 4) != 0 )
      return 1;
    v12 = *(_QWORD *)(v10 + 16);
    v13 = *(_QWORD *)(v12 + 32);
    for ( i = v13 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF); i != v13; v13 += 40 )
    {
      if ( *(_BYTE *)v13 == 12 )
      {
        v15 = ((*(_DWORD *)(*(_QWORD *)(v13 + 24) + v9) >> v8) & 1) == 0;
        if ( ((*(_DWORD *)(*(_QWORD *)(v13 + 24) + v9) >> v8) & 1) == 0 )
          return v15;
      }
      else if ( !*(_BYTE *)v13
             && (*(_BYTE *)(v13 + 3) & 0x10) != 0
             && a4 == *(_DWORD *)(v13 + 8)
             && (v11 || (*(_BYTE *)(v13 + 4) & 4) != 0 || (unsigned int)*(unsigned __int16 *)(v12 + 68) - 1 <= 1) )
      {
        return 1;
      }
    }
    v18 = v8;
    v17 = sub_220EF30(v4);
    v8 = v18;
    v4 = v17;
  }
  while ( a3 != v17 );
  return 0;
}
