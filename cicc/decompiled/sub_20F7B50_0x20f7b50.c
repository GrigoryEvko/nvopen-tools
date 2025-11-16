// Function: sub_20F7B50
// Address: 0x20f7b50
//
bool __fastcall sub_20F7B50(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r9d
  __int64 v6; // r10
  unsigned int v7; // edx
  _WORD *v8; // rsi
  _WORD *v9; // rcx
  unsigned __int16 v10; // ax
  __int64 v11; // rdx
  int v12; // edi
  _DWORD *i; // rdx

  if ( !a3 )
    BUG();
  v5 = a1[14];
  v6 = *a1;
  v7 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24 * v6 + 16);
  v8 = (_WORD *)(*(_QWORD *)(a3 + 56) + 2LL * (v7 >> 4));
  v9 = v8 + 1;
  v10 = *v8 + v6 * (v7 & 0xF);
  if ( v5 )
  {
    v11 = *((_QWORD *)a1 + 6);
    v12 = 0;
    for ( i = (_DWORD *)(v11 + 88); *i == *(_DWORD *)(a2 + 216LL * v10); i += 28 )
    {
      ++v12;
      v10 += *v9;
      if ( !*v9 )
        return v5 == v12;
      ++v9;
      if ( v5 == v12 )
        return 0;
    }
  }
  return 0;
}
