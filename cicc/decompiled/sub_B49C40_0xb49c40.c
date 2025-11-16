// Function: sub_B49C40
// Address: 0xb49c40
//
__int64 __fastcall sub_B49C40(__int64 a1, unsigned int a2, char a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // rax
  int v10; // eax

  v4 = sub_B49B80(a1, a2, 43);
  if ( (_BYTE)v4 )
  {
    v5 = v4;
    if ( a3 || (unsigned __int8)sub_B49B80(a1, a2, 40) )
      return v5;
  }
  v5 = 0;
  if ( !sub_A745B0((_QWORD *)(a1 + 72), a2) )
    return v5;
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (a2 - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  v8 = *(_DWORD *)(v7 + 8);
  v9 = sub_B491C0(a1);
  LOBYTE(v10) = sub_B2F070(v9, v8 >> 8);
  return v10 ^ 1u;
}
