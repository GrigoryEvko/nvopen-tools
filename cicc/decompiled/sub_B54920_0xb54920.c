// Function: sub_B54920
// Address: 0xb54920
//
__int64 __fastcall sub_B54920(__int64 a1, int a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // rax
  __int64 *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 32LL * v2;
  v4 = v2 - 1;
  v5 = v3 + 32LL * (unsigned int)(a2 + 1);
  v6 = (__int64 *)(v3 + 32 * v4);
  v7 = *v6;
  if ( *(_QWORD *)v5 )
  {
    v8 = *(_QWORD *)(v5 + 8);
    **(_QWORD **)(v5 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v5 + 16);
  }
  *(_QWORD *)v5 = v7;
  if ( v7 )
  {
    v9 = *(_QWORD *)(v7 + 16);
    *(_QWORD *)(v5 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v5 + 8;
    *(_QWORD *)(v5 + 16) = v7 + 16;
    *(_QWORD *)(v7 + 16) = v5;
  }
  if ( *v6 )
  {
    v10 = v6[1];
    *(_QWORD *)v6[2] = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v6[2];
  }
  *v6 = 0;
  result = v4 & 0x7FFFFFF | *(_DWORD *)(a1 + 4) & 0xF8000000;
  *(_DWORD *)(a1 + 4) = result;
  return result;
}
