// Function: sub_223FB20
// Address: 0x223fb20
//
__int64 __fastcall sub_223FB20(__int64 a1, __int64 a2, int a3, unsigned int a4)
{
  int v6; // r9d
  bool v7; // dl
  unsigned int v8; // eax
  unsigned __int8 v9; // r11
  int v10; // r10d
  char v11; // cl
  unsigned __int8 v12; // dl
  __int64 v13; // rax
  char v15; // cl
  unsigned __int64 v16; // r10
  __int64 v17; // rbx

  v6 = *(_DWORD *)(a1 + 64) & 8;
  v7 = (v6 & a4) != 0;
  v8 = (a4 & *(_DWORD *)(a1 + 64)) >> 4;
  v9 = (a3 != 1) & v8 & v7;
  v10 = (a4 >> 4) ^ 1;
  v11 = v8 & ((a4 >> 3) ^ 1) & 1;
  v12 = v10 & v7;
  if ( v12 )
  {
    v13 = *(_QWORD *)(a1 + 8);
    if ( !v13 && a2 )
      return -1;
    v15 = v9 | v11;
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 32);
    if ( !v13 && a2 )
      return -1;
    v15 = v9 | v11;
    if ( !v15 )
      return -1;
  }
  v16 = *(_QWORD *)(a1 + 40);
  if ( v16 && v16 > *(_QWORD *)(a1 + 24) )
  {
    if ( !v6 )
    {
      *(_QWORD *)(a1 + 8) = v16;
      *(_QWORD *)(a1 + 16) = v16;
    }
    *(_QWORD *)(a1 + 24) = v16;
  }
  if ( a3 == 1 )
  {
    v17 = v16 - v13 + a2;
    a2 += *(_QWORD *)(a1 + 16) - v13;
  }
  else
  {
    v17 = a2;
    if ( a3 == 2 )
    {
      a2 += *(_QWORD *)(a1 + 24) - v13;
      v17 = a2;
    }
  }
  if ( v12 | v9 && a2 >= 0 && *(_QWORD *)(a1 + 24) - v13 >= a2 )
    *(_QWORD *)(a1 + 16) = a2 + *(_QWORD *)(a1 + 8);
  else
    a2 = -1;
  if ( v17 >= 0 && v15 && *(_QWORD *)(a1 + 24) - v13 >= v17 )
  {
    sub_223FAE0((_QWORD *)a1, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 48), v17);
    return v17;
  }
  return a2;
}
