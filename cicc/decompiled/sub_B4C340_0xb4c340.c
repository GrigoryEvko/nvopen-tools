// Function: sub_B4C340
// Address: 0xb4c340
//
__int64 __fastcall sub_B4C340(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // ecx
  bool v4; // zf
  __int64 *v5; // rax
  int v6; // r10d
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // r9
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rsi

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 10, 0x8000000u, 0, 0);
  v2 = 0;
  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = (*(_BYTE *)(a2 + 2) & 1) == 0;
  *(_DWORD *)(a1 + 4) = v3 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  v5 = *(__int64 **)(a2 - 8);
  if ( !v4 )
    v2 = v5[4];
  sub_B4C1E0(a1, *v5, v2, v3);
  v6 = *(_DWORD *)(a1 + 72);
  v7 = v6 & 0x7FFFFFF;
  result = (unsigned int)v7 | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  if ( (result & 0x40000000) != 0 )
    v9 = *(_QWORD *)(a1 - 8);
  else
    v9 = a1 - 32 * v7;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v10 = *(_QWORD *)(a2 - 8);
  }
  else
  {
    result = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v10 = a2 - result;
  }
  v11 = 1;
  if ( v6 != 1 )
  {
    do
    {
      v12 = 32LL * v11;
      result = v9 + v12;
      v13 = *(_QWORD *)(v10 + v12);
      if ( *(_QWORD *)result )
      {
        v14 = *(_QWORD *)(result + 8);
        **(_QWORD **)(result + 16) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(result + 16);
      }
      *(_QWORD *)result = v13;
      if ( v13 )
      {
        v15 = *(_QWORD *)(v13 + 16);
        *(_QWORD *)(result + 8) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = result + 8;
        *(_QWORD *)(result + 16) = v13 + 16;
        *(_QWORD *)(v13 + 16) = result;
      }
      ++v11;
    }
    while ( v6 != v11 );
  }
  return result;
}
