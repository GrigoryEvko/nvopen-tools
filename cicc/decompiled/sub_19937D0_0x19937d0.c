// Function: sub_19937D0
// Address: 0x19937d0
//
char __fastcall sub_19937D0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ecx
  char result; // al
  __int64 v6; // r15
  __int64 v7; // r13
  char v8; // [rsp+Fh] [rbp-31h]

  v4 = *(_DWORD *)(a2 + 32);
  if ( v4 != 2 )
    return sub_1993620(
             a1,
             *(_QWORD *)(a2 + 712),
             *(_QWORD *)(a2 + 720),
             v4,
             *(_QWORD *)(a2 + 40),
             *(_DWORD *)(a2 + 48),
             *(_QWORD *)a3,
             *(_QWORD *)(a3 + 8),
             *(_BYTE *)(a3 + 16),
             *(_QWORD *)(a3 + 24));
  v8 = sub_14A2CC0((__int64)a1);
  if ( !v8 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    return sub_1993620(
             a1,
             *(_QWORD *)(a2 + 712),
             *(_QWORD *)(a2 + 720),
             v4,
             *(_QWORD *)(a2 + 40),
             *(_DWORD *)(a2 + 48),
             *(_QWORD *)a3,
             *(_QWORD *)(a3 + 8),
             *(_BYTE *)(a3 + 16),
             *(_QWORD *)(a3 + 24));
  }
  v6 = *(_QWORD *)(a2 + 56);
  v7 = v6 + 80LL * *(unsigned int *)(a2 + 64);
  if ( v7 == v6 )
    return v8;
  while ( 1 )
  {
    result = sub_14A2A90(
               a1,
               *(_QWORD *)(a2 + 40),
               *(_QWORD *)a3,
               *(_QWORD *)(a3 + 8) + *(_QWORD *)(v6 + 72),
               *(_BYTE *)(a3 + 16),
               *(_QWORD *)(a3 + 24));
    if ( !result )
      break;
    v6 += 80;
    if ( v7 == v6 )
      return v8;
  }
  return result;
}
