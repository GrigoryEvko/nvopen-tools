// Function: sub_251C440
// Address: 0x251c440
//
__int64 __fastcall sub_251C440(__int64 a1, __int64 a2, _QWORD *a3, _BYTE *a4, char a5, int a6)
{
  __int64 v11; // r15
  unsigned __int64 v12; // rdi
  unsigned __int8 v13; // al
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rcx
  int v17; // edx
  unsigned int v18; // eax
  __int64 v19; // rsi
  int v20; // r10d
  int v21; // [rsp-40h] [rbp-40h]
  char v22; // [rsp-3Ch] [rbp-3Ch]

  if ( !*(_BYTE *)(a1 + 4300) )
    return 0;
  v11 = *(_QWORD *)(a1 + 200);
  v12 = *(_QWORD *)(a2 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a2 + 72) & 3LL) == 3 )
    v12 = *(_QWORD *)(v12 + 24);
  v13 = *(_BYTE *)v12;
  if ( *(_BYTE *)v12 )
  {
    if ( v13 == 22 )
    {
      v12 = *(_QWORD *)(v12 + 24);
    }
    else if ( v13 <= 0x1Cu )
    {
      v12 = 0;
    }
    else
    {
      v21 = a6;
      v22 = a5;
      v14 = sub_B43CB0(v12);
      a5 = v22;
      a6 = v21;
      v12 = v14;
    }
  }
  v15 = *(_DWORD *)(v11 + 24);
  v16 = *(_QWORD *)(v11 + 8);
  if ( v15 )
  {
    v17 = v15 - 1;
    v18 = (v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v19 = *(_QWORD *)(v16 + 8LL * v18);
    if ( v12 == v19 )
      return sub_251C230(a1, (__int64 *)(a2 + 72), a2, a3, a4, a5, a6);
    v20 = 1;
    while ( v19 != -4096 )
    {
      v18 = v17 & (v20 + v18);
      v19 = *(_QWORD *)(v16 + 8LL * v18);
      if ( v12 == v19 )
        return sub_251C230(a1, (__int64 *)(a2 + 72), a2, a3, a4, a5, a6);
      ++v20;
    }
  }
  return 0;
}
