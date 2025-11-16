// Function: sub_16A1240
// Address: 0x16a1240
//
__int64 __fastcall sub_16A1240(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v6; // r12d
  void *v8; // rax
  _QWORD *v9; // r13
  void *v10; // rsi
  __int64 v11; // rax
  bool v12; // al
  __int64 v13; // r13
  char v14; // di
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  char v18; // al

  v6 = sub_16A1030(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8), a3, a4, a5);
  if ( v6 == 1 )
  {
    v6 = sub_16A1030(*(_QWORD *)(a1 + 8) + 32LL, *(_QWORD *)(a2 + 8) + 32LL, a3, a4, a5);
    if ( (v6 & 0xFFFFFFFD) == 0 )
    {
      v8 = sub_16982C0();
      v9 = *(_QWORD **)(a1 + 8);
      v10 = v8;
      v11 = (__int64)(v9 + 1);
      if ( (void *)v9[1] == v10 )
        v11 = v9[2] + 8LL;
      v12 = (*(_BYTE *)(v11 + 18) & 8) != 0;
      if ( (void *)v9[5] == v10 )
        v13 = v9[6] + 8LL;
      else
        v13 = (__int64)(v9 + 5);
      v14 = v12 ^ ((*(_BYTE *)(v13 + 18) & 8) != 0);
      v15 = *(_QWORD **)(a2 + 8);
      v16 = (__int64)(v15 + 1);
      if ( (void *)v15[1] == v10 )
        v16 = v15[2] + 8LL;
      if ( (void *)v15[5] == v10 )
        v17 = v15[6] + 8LL;
      else
        v17 = (__int64)(v15 + 5);
      v18 = ((*(_BYTE *)(v16 + 18) & 8) != 0) ^ ((*(_BYTE *)(v17 + 18) & 8) != 0);
      if ( v18 )
      {
        if ( v14 )
          return 2 - v6;
      }
      else if ( v14 )
      {
        return 0;
      }
      if ( v18 )
        return 2;
    }
  }
  return v6;
}
