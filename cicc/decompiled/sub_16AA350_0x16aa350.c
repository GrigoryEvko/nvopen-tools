// Function: sub_16AA350
// Address: 0x16aa350
//
__int64 __fastcall sub_16AA350(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned int v9; // r15d
  bool v10; // dl
  int v11; // eax
  unsigned int v13; // r15d

  v7 = *(_DWORD *)(a2 + 8);
  v8 = *(_QWORD *)a2;
  v9 = v7 - 1;
  if ( v7 <= 0x40 )
  {
    v10 = 0;
    if ( v8 != 1LL << v9 )
      goto LABEL_4;
    goto LABEL_6;
  }
  v10 = 0;
  if ( (*(_QWORD *)(v8 + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
  {
    v11 = sub_16A58A0(a2);
    v10 = 0;
    if ( v11 == v9 )
    {
LABEL_6:
      v13 = *(_DWORD *)(a3 + 8);
      if ( v13 <= 0x40 )
        v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *(_QWORD *)a3;
      else
        v10 = v13 == (unsigned int)sub_16A58F0(a3);
    }
  }
LABEL_4:
  *a4 = v10;
  sub_16A9F90(a1, a2, a3);
  return a1;
}
