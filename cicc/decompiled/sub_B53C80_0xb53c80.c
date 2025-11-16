// Function: sub_B53C80
// Address: 0xb53c80
//
__int64 __fastcall sub_B53C80(__int64 a1, __int64 a2, int a3)
{
  unsigned int v4; // ecx
  __int64 v5; // r10
  __int64 v6; // r11
  int v7; // ebx
  __int64 *v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rsi

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 32LL * v4;
  v6 = v4 - 2;
  v7 = 2 * a3 + 4;
  v8 = (__int64 *)(v5 + 32LL * (v4 - 1));
  v9 = (__int64 *)(v5 + 32 * v6);
  if ( v7 != v4 )
  {
    v13 = *v9;
    v14 = v5 + 32LL * (unsigned int)(2 * a3 + 2);
    if ( *(_QWORD *)v14 )
    {
      v15 = *(_QWORD *)(v14 + 8);
      **(_QWORD **)(v14 + 16) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v14 + 16);
    }
    *(_QWORD *)v14 = v13;
    if ( v13 )
    {
      v16 = *(_QWORD *)(v13 + 16);
      *(_QWORD *)(v14 + 8) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = v14 + 8;
      *(_QWORD *)(v14 + 16) = v13 + 16;
      *(_QWORD *)(v13 + 16) = v14;
    }
    v17 = *v8;
    v18 = v5 + 32LL * (unsigned int)(v7 - 1);
    if ( *(_QWORD *)v18 )
    {
      v19 = *(_QWORD *)(v18 + 8);
      **(_QWORD **)(v18 + 16) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
    }
    *(_QWORD *)v18 = v17;
    if ( v17 )
    {
      v20 = *(_QWORD *)(v17 + 16);
      *(_QWORD *)(v18 + 8) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = v18 + 8;
      *(_QWORD *)(v18 + 16) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v18;
    }
  }
  if ( *v9 )
  {
    v10 = v9[1];
    *(_QWORD *)v9[2] = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v9[2];
  }
  *v9 = 0;
  if ( *v8 )
  {
    v11 = v8[1];
    *(_QWORD *)v8[2] = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = v8[2];
  }
  *v8 = 0;
  *(_DWORD *)(a1 + 4) = v6 & 0x7FFFFFF | *(_DWORD *)(a1 + 4) & 0xF8000000;
  return a1;
}
