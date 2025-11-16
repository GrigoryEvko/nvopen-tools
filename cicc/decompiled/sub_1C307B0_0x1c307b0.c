// Function: sub_1C307B0
// Address: 0x1c307b0
//
char __fastcall sub_1C307B0(__int64 a1)
{
  __int64 v2; // rdi
  char result; // al
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax

  v2 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v2 + 16) )
    return 0;
  result = sub_1560180(v2 + 112, 47);
  if ( result )
    return result;
  v4 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v4 + 16) || (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
    return 0;
  v5 = *(_DWORD *)(v4 + 36);
  if ( v5 != 4086 )
  {
    if ( v5 == 4350 )
    {
      v6 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v7 = *(_QWORD **)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
        v7 = (_QWORD *)*v7;
      return (unsigned int)((_DWORD)v7 - 82) > 1;
    }
    if ( sub_1C30440(v5) )
    {
      v8 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v9 = *(_QWORD **)(v8 + 24);
      if ( *(_DWORD *)(v8 + 32) > 0x40u )
        v9 = (_QWORD *)*v9;
      return ((BYTE1(v9) >> 5) ^ 1) & 1;
    }
    if ( sub_1C30480(v5) )
    {
      v12 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v13 = *(_QWORD **)(v12 + 24);
      if ( *(_DWORD *)(v12 + 32) > 0x40u )
        v13 = (_QWORD *)*v13;
      return ((BYTE4(v13) >> 4) ^ 1) & 1;
    }
    if ( sub_1C304C0(v5) )
    {
      v14 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v15 = *(_QWORD **)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) > 0x40u )
        v15 = (_QWORD *)*v15;
      return ((BYTE1(v15) >> 4) ^ 1) & 1;
    }
    if ( sub_1C304F0(v5) )
    {
      v16 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v17 = *(_QWORD **)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v17 = (_QWORD *)*v17;
      return (((unsigned __int8)v17 >> 3) ^ 1) & 1;
    }
    return 0;
  }
  v10 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  return (((unsigned __int8)v11 >> 2) ^ 1) & 1;
}
