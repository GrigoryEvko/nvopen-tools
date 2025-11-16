// Function: sub_CF49B0
// Address: 0xcf49b0
//
char __fastcall sub_CF49B0(unsigned __int8 *a1, unsigned int a2, int a3)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  char result; // al
  unsigned int *v14; // rdx

  v4 = *a1;
  if ( v4 == 40 )
  {
    v5 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v5 = 0;
    if ( v4 != 85 )
    {
      v5 = 64;
      if ( v4 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_10;
  v6 = sub_BD2BC0((__int64)a1);
  v8 = v6 + v7;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v8 >> 4) )
LABEL_20:
      BUG();
LABEL_10:
    v12 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v8 - sub_BD2BC0((__int64)a1)) >> 4) )
    goto LABEL_10;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_20;
  v9 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v10 = sub_BD2BC0((__int64)a1);
  v12 = 32LL * (unsigned int)(*(_DWORD *)(v10 + v11 - 4) - v9);
LABEL_11:
  if ( a2 < (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v5 - v12) >> 5) )
    return sub_B49B80((__int64)a1, a2, a3);
  v14 = (unsigned int *)sub_B49810((__int64)a1, a2);
  result = a3 == 51 && *(_DWORD *)(*(_QWORD *)v14 + 8LL) == 0;
  if ( result )
    return *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a1[32 * (v14[2] - (unsigned __int64)(*((_DWORD *)a1 + 1) & 0x7FFFFFF))
                                               + 32 * (a2 - v14[2])]
                                + 8LL)
                    + 8LL) == 14;
  return result;
}
