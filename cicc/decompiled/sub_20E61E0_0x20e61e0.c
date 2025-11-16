// Function: sub_20E61E0
// Address: 0x20e61e0
//
__int64 __fastcall sub_20E61E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r9
  __int64 v6; // r12
  int v8; // ecx
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // di
  __int64 v12; // r8
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rsi
  _BOOL4 v16; // edx
  __int64 v18; // rax
  int v19; // [rsp-2Ch] [rbp-2Ch]

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
    v13 = *(_DWORD *)(v12 + 40);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v12 + 32);
      v15 = v14 + 40LL * (unsigned int)(v13 - 1) + 40;
      do
      {
        if ( *(_BYTE *)v14 == 12 )
        {
          v16 = ((*(_DWORD *)(*(_QWORD *)(v14 + 24) + v9) >> v8) & 1) == 0;
          if ( ((*(_DWORD *)(*(_QWORD *)(v14 + 24) + v9) >> v8) & 1) == 0 )
            return v16;
        }
        else if ( !*(_BYTE *)v14
               && (*(_BYTE *)(v14 + 3) & 0x10) != 0
               && a4 == *(_DWORD *)(v14 + 8)
               && (v11 || (*(_BYTE *)(v14 + 4) & 4) != 0 || **(_WORD **)(v12 + 16) == 1) )
        {
          return 1;
        }
        v14 += 40;
      }
      while ( v15 != v14 );
    }
    v19 = v8;
    v18 = sub_220EF30(v4);
    v8 = v19;
    v4 = v18;
  }
  while ( a3 != v18 );
  return 0;
}
