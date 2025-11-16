// Function: sub_B53E30
// Address: 0xb53e30
//
__int64 __fastcall sub_B53E30(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ebx
  unsigned int v5; // r12d
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v5 = v4 + 2;
  v6 = v4 >> 1;
  if ( v5 > *(_DWORD *)(a1 + 72) )
    sub_B53E10(a1);
  v7 = 2 * v6;
  v8 = *(_QWORD *)(a1 - 8) + 32 * v7;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | v5 & 0x7FFFFFF;
  if ( *(_QWORD *)v8 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    **(_QWORD **)(v8 + 16) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
  }
  *(_QWORD *)v8 = a2;
  if ( a2 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v8 + 8) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v8 + 8;
    *(_QWORD *)(v8 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v8;
  }
  result = *(_QWORD *)(a1 - 8) + 32LL * (unsigned int)(v7 + 1);
  if ( *(_QWORD *)result )
  {
    v12 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(result + 8) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = result + 8;
    *(_QWORD *)(result + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = result;
  }
  return result;
}
