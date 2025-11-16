// Function: sub_372B3C0
// Address: 0x372b3c0
//
__int64 __fastcall sub_372B3C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // rax
  unsigned __int64 v5; // rcx
  __int16 v6; // ax
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax

  v2 = *(__int64 **)a2;
  v3 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( (__int64 *)v3 == v2 )
    return 0;
  while ( 1 )
  {
    v4 = *v2;
    if ( (*v2 & 4) == 0 )
      break;
LABEL_11:
    v2 += 2;
    if ( (__int64 *)v3 == v2 )
      return 0;
  }
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(_WORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 68);
  if ( v6 == 14 )
  {
    v10 = *(_QWORD *)(v5 + 32);
    v9 = v10 + 40;
LABEL_8:
    while ( v9 != v10 )
    {
      if ( !*(_BYTE *)v10 && !*(_DWORD *)(v10 + 8) )
        goto LABEL_11;
      v10 += 40;
    }
  }
  else if ( v6 == 15 )
  {
    v8 = *(_QWORD *)(v5 + 32);
    v9 = v8 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
    v10 = v8 + 80;
    goto LABEL_8;
  }
  return 1;
}
