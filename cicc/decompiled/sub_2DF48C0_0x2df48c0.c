// Function: sub_2DF48C0
// Address: 0x2df48c0
//
bool __fastcall sub_2DF48C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // edx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // r14
  __int64 v13; // r13

  v5 = *(unsigned int *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 8) + 16 * v5 - 16;
  v7 = *(_DWORD *)(v6 + 12);
  if ( *(_DWORD *)(*(_QWORD *)a1 + 160LL) )
  {
    if ( !v7 )
    {
      v11 = sub_F03A30((__int64 *)(a1 + 8), (int)v5 - 1);
      if ( !v11 )
        return 0;
      v12 = v11 & 0xFFFFFFFFFFFFFFC0LL;
      v13 = v11 & 0x3F;
      if ( !sub_2DF4840((v11 & 0xFFFFFFFFFFFFFFC0LL) + 24 * v13 + 64, a3) )
        return 0;
      return *(_QWORD *)(v12 + 16 * v13 + 8) == a2;
    }
  }
  else if ( !v7 )
  {
    return 0;
  }
  v9 = *(_QWORD *)v6;
  v10 = (unsigned int)(v7 - 1);
  if ( !sub_2DF4840(*(_QWORD *)v6 + 24 * v10 + 64, a3) )
    return 0;
  return *(_QWORD *)(v9 + 16 * v10 + 8) == a2;
}
