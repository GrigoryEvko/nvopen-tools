// Function: sub_B53AE0
// Address: 0xb53ae0
//
__int64 __fastcall sub_B53AE0(__int64 a1, __int64 a2)
{
  unsigned int v4; // r8d
  unsigned int v5; // eax
  __int64 v6; // rdi
  __int64 v7; // rsi
  unsigned int i; // edx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r9
  __int64 v18; // r9
  __int64 result; // rax

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 3, 0x8000000u, 0, 0);
  sub_B53980(a1, **(_QWORD **)(a2 - 8), *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL), *(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = v4 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = v5;
  if ( (v5 & 0x40000000) != 0 )
    v6 = *(_QWORD *)(a1 - 8);
  else
    v6 = a1 - 32LL * v4;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * v4;
  if ( v4 != 2 )
  {
    for ( i = 2; i != v4; i += 2 )
    {
      v9 = 32LL * i;
      v10 = v6 + v9;
      v11 = *(_QWORD *)(v7 + v9);
      if ( *(_QWORD *)v10 )
      {
        v12 = *(_QWORD *)(v10 + 8);
        **(_QWORD **)(v10 + 16) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v10 + 16);
      }
      *(_QWORD *)v10 = v11;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v10 + 8) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = v10 + 8;
        *(_QWORD *)(v10 + 16) = v11 + 16;
        *(_QWORD *)(v11 + 16) = v10;
      }
      v14 = 32LL * (i + 1);
      v15 = v6 + v14;
      v16 = *(_QWORD *)(v7 + v14);
      if ( *(_QWORD *)v15 )
      {
        v17 = *(_QWORD *)(v15 + 8);
        **(_QWORD **)(v15 + 16) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v15 + 16);
      }
      *(_QWORD *)v15 = v16;
      if ( v16 )
      {
        v18 = *(_QWORD *)(v16 + 16);
        *(_QWORD *)(v15 + 8) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = v15 + 8;
        *(_QWORD *)(v15 + 16) = v16 + 16;
        *(_QWORD *)(v16 + 16) = v15;
      }
    }
  }
  result = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1u;
  *(_BYTE *)(a1 + 1) = *(_BYTE *)(a2 + 1) & 0xFE | *(_BYTE *)(a1 + 1) & 1;
  return result;
}
