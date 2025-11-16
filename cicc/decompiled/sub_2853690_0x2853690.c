// Function: sub_2853690
// Address: 0x2853690
//
char __fastcall sub_2853690(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r9d
  char result; // al
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rcx
  char v9; // al
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  char v12; // [rsp+Fh] [rbp-31h]

  v4 = *(_DWORD *)(a2 + 32);
  if ( v4 != 2 )
    return sub_2850670(
             a1,
             *(_QWORD *)(a2 + 712),
             *(_BYTE *)(a2 + 720),
             *(_QWORD *)(a2 + 728),
             *(_BYTE *)(a2 + 736),
             v4,
             *(_QWORD *)(a2 + 40),
             *(unsigned int *)(a2 + 48),
             *(_QWORD *)a3,
             *(_QWORD *)(a3 + 8),
             *(_BYTE *)(a3 + 16),
             *(_BYTE *)(a3 + 24),
             *(_QWORD *)(a3 + 32));
  v12 = sub_DFA830((__int64)a1);
  if ( !v12 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    return sub_2850670(
             a1,
             *(_QWORD *)(a2 + 712),
             *(_BYTE *)(a2 + 720),
             *(_QWORD *)(a2 + 728),
             *(_BYTE *)(a2 + 736),
             v4,
             *(_QWORD *)(a2 + 40),
             *(unsigned int *)(a2 + 48),
             *(_QWORD *)a3,
             *(_QWORD *)(a3 + 8),
             *(_BYTE *)(a3 + 16),
             *(_BYTE *)(a3 + 24),
             *(_QWORD *)(a3 + 32));
  }
  v6 = *(_QWORD *)(a2 + 56);
  v7 = v6 + 80LL * *(unsigned int *)(a2 + 64);
  if ( v7 == v6 )
    return v12;
  while ( 1 )
  {
    v8 = *(_QWORD *)(v6 + 64);
    v9 = *(_BYTE *)(a3 + 16);
    v10 = v8 + *(_QWORD *)(a3 + 8);
    if ( v8 )
      v9 = *(_BYTE *)(v6 + 72);
    v11 = 0;
    if ( !v9 )
      v11 = v10;
    result = sub_DFA150(a1, *(_QWORD *)(a2 + 40), *(_QWORD *)a3, v11, *(_BYTE *)(a3 + 24), *(_QWORD *)(a3 + 32));
    if ( !result )
      break;
    v6 += 80;
    if ( v7 == v6 )
      return v12;
  }
  return result;
}
