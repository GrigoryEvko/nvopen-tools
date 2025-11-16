// Function: sub_2A10B40
// Address: 0x2a10b40
//
__int64 __fastcall sub_2A10B40(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  v1 = sub_D4B130(a1);
  v3 = sub_AA5930(**(_QWORD **)(a1 + 32));
  if ( v3 == v2 )
    return 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(v3 - 8);
    v5 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v3 + 4) & 0x7FFFFFF) != 0 )
      break;
LABEL_7:
    if ( **(_BYTE **)(v4 + v5) == 17 )
      return 1;
LABEL_8:
    v7 = *(_QWORD *)(v3 + 32);
    if ( !v7 )
      BUG();
    v3 = 0;
    if ( *(_BYTE *)(v7 - 24) == 84 )
      v3 = v7 - 24;
    if ( v2 == v3 )
      return 0;
  }
  v6 = 0;
  do
  {
    if ( v1 == *(_QWORD *)(v4 + 32LL * *(unsigned int *)(v3 + 72) + 8 * v6) )
    {
      v5 = 32 * v6;
      goto LABEL_7;
    }
    ++v6;
  }
  while ( (*(_DWORD *)(v3 + 4) & 0x7FFFFFF) != (_DWORD)v6 );
  if ( **(_BYTE **)(v4 + 0x1FFFFFFFE0LL) != 17 )
    goto LABEL_8;
  return 1;
}
