// Function: sub_1759270
// Address: 0x1759270
//
__int64 __fastcall sub_1759270(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rdi
  __int64 v7; // rbx
  unsigned __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // r15
  unsigned __int8 v16; // al
  unsigned __int8 v17; // [rsp+Fh] [rbp-31h]

  v5 = *(_QWORD *)(a2 + 40);
  if ( !v5 )
    return 0;
  v7 = a4;
  v8 = sub_157EBA0(v5);
  v9 = v8;
  if ( !v8 )
    return 0;
  if ( *(_BYTE *)(v8 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v8 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v11 = *(_QWORD *)(v8 - 72);
  if ( *(_BYTE *)(v11 + 16) != 75 )
    return 0;
  v12 = *(_QWORD *)(v11 - 48);
  if ( a2 != v12 || !v12 )
  {
    v13 = *(_QWORD *)(v11 - 24);
    if ( !v13 || a2 != v13 )
      return 0;
  }
  v14 = *(unsigned __int16 *)(a3 + 18);
  BYTE1(v14) &= ~0x80u;
  if ( v14 != 32 )
    return 0;
  v15 = sub_15F4DF0(v9, 1u);
  if ( !sub_157F0B0(v15) )
    return 0;
  v16 = sub_17591E0(a1, a2, (_QWORD *)a3, v15);
  if ( !v16 )
    return 0;
  v17 = v16;
  sub_1648F20(a2, *(_QWORD *)(a2 + 24 * v7 - 72), *(_QWORD *)(a2 + 40));
  return v17;
}
