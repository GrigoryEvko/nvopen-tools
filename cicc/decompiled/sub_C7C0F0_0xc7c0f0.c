// Function: sub_C7C0F0
// Address: 0xc7c0f0
//
__int64 __fastcall sub_C7C0F0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r14d
  unsigned int v4; // r15d
  unsigned __int64 v5; // rdx
  unsigned int v6; // eax
  unsigned int v7; // ebx
  unsigned int v8; // eax
  unsigned int v9; // esi
  int v10; // eax
  unsigned int v15; // eax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 8);
  v21 = v3;
  if ( v3 <= 0x40 )
  {
    v20 = 0;
    v4 = v3;
LABEL_3:
    v5 = *(_QWORD *)a2;
    goto LABEL_4;
  }
  sub_C43690((__int64)&v20, 0, 0);
  v4 = *(_DWORD *)(a2 + 8);
  v23 = v4;
  if ( v4 <= 0x40 )
    goto LABEL_3;
  sub_C43780((__int64)&v22, (const void **)a2);
  v4 = v23;
  v5 = v22;
LABEL_4:
  v6 = v21;
  *(_DWORD *)(a1 + 8) = v4;
  *(_QWORD *)a1 = v5;
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 16) = v20;
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 <= 0x40 )
  {
    _RCX = *(_QWORD *)(a2 + 16);
    v15 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( _RCX )
      v15 = _RSI;
    if ( v7 > v15 )
      v7 = v15;
  }
  else
  {
    v19 = v5;
    v8 = sub_C44590(a2 + 16);
    v5 = v19;
    v7 = v8;
  }
  v9 = v7 + 1;
  if ( v7 + 1 > v3 )
    v9 = v3;
  if ( v4 != v9 )
  {
    if ( v9 > 0x3F || v4 > 0x40 )
      sub_C43C90((_QWORD *)a1, v9, v4);
    else
      *(_QWORD *)a1 = v5 | (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 - (unsigned __int8)v4 + 64) << v9);
  }
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
  {
    v10 = sub_C445E0(a2);
  }
  else
  {
    v10 = 64;
    _RDX = ~*(_QWORD *)a2;
    __asm { tzcnt   rcx, rdx }
    if ( *(_QWORD *)a2 != -1 )
      v10 = _RCX;
  }
  if ( v7 == v10 && v3 > v7 )
  {
    v17 = *(_QWORD *)(a1 + 16);
    v18 = 1LL << v7;
    if ( *(_DWORD *)(a1 + 24) > 0x40u )
      *(_QWORD *)(v17 + 8LL * (v7 >> 6)) |= v18;
    else
      *(_QWORD *)(a1 + 16) = v17 | v18;
  }
  return a1;
}
