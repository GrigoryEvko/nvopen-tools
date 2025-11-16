// Function: sub_1171EF0
// Address: 0x1171ef0
//
__int64 __fastcall sub_1171EF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v5; // r14
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 )
  {
    while ( **(_BYTE **)(v3 + 24) == 76 )
    {
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        goto LABEL_5;
    }
    return 0;
  }
LABEL_5:
  v5 = 0;
  v6 = 0;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
    return 0;
  do
  {
    v13 = 32LL * v6;
    v14 = sub_F0C930(a1, *(_QWORD *)(*(_QWORD *)(a2 - 8) + v13));
    if ( v14 )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v7 = v13 + *(_QWORD *)(a2 - 8);
      else
        v7 = a2 + v13 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v15 = *(_QWORD *)v7;
      if ( *(_QWORD *)v7 )
      {
        v8 = *(_QWORD *)(v7 + 8);
        **(_QWORD **)(v7 + 16) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
      }
      *(_QWORD *)v7 = v14;
      v9 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v7 + 8) = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = v7 + 8;
      *(_QWORD *)(v7 + 16) = v14 + 16;
      *(_QWORD *)(v14 + 16) = v7;
      if ( *(_BYTE *)v15 > 0x1Cu )
      {
        v10 = *(_QWORD *)(a1 + 40);
        v16[0] = v15;
        v11 = v10 + 2096;
        sub_11715E0(v10 + 2096, v16);
        v12 = *(_QWORD *)(v15 + 16);
        if ( v12 )
        {
          if ( !*(_QWORD *)(v12 + 8) )
          {
            v16[0] = *(_QWORD *)(v12 + 24);
            sub_11715E0(v11, v16);
          }
        }
      }
      v5 = 1;
    }
    ++v6;
  }
  while ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != v6 );
  if ( !v5 )
    return 0;
  return a2;
}
