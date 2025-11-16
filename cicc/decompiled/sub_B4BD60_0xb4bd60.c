// Function: sub_B4BD60
// Address: 0xb4bd60
//
__int64 __fastcall sub_B4BD60(__int64 a1, __int64 a2, unsigned int a3)
{
  int v3; // edx
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdx

  sub_B44260(a1, *(_QWORD *)(a2 + 8), 8, a3, 0, 0);
  v3 = *(_DWORD *)(a1 + 4);
  *(_WORD *)(a1 + 2) = *(_WORD *)(a2 + 2) & 0x7FFF | *(_WORD *)(a1 + 2) & 0x8000;
  result = a1 - 32LL * (v3 & 0x7FFFFFF);
  v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)result )
  {
    v6 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = v5;
  if ( v5 )
  {
    v7 = *(_QWORD *)(v5 + 16);
    *(_QWORD *)(result + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = result + 8;
    *(_QWORD *)(result + 16) = v5 + 16;
    *(_QWORD *)(v5 + 16) = result;
  }
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    v8 = 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + a1;
    result = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( *(_QWORD *)v8 )
    {
      v9 = *(_QWORD *)(v8 + 8);
      **(_QWORD **)(v8 + 16) = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
    }
    *(_QWORD *)v8 = result;
    if ( result )
    {
      v10 = *(_QWORD *)(result + 16);
      *(_QWORD *)(v8 + 8) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = v8 + 8;
      *(_QWORD *)(v8 + 16) = result + 16;
      *(_QWORD *)(result + 16) = v8;
    }
  }
  return result;
}
