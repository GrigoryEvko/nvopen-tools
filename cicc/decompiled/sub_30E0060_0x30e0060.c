// Function: sub_30E0060
// Address: 0x30e0060
//
__int64 __fastcall sub_30E0060(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4)
{
  int v6; // edx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 result; // rax

  v6 = *a3;
  if ( v6 == 40 )
  {
    v7 = 32LL * (unsigned int)sub_B491D0((__int64)a3);
  }
  else
  {
    v7 = 0;
    if ( v6 != 85 )
    {
      v7 = 64;
      if ( v6 != 34 )
LABEL_17:
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_10;
  v8 = sub_BD2BC0((__int64)a3);
  v10 = v8 + v9;
  if ( (a3[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v10 >> 4) )
      goto LABEL_17;
    goto LABEL_10;
  }
  if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)a3)) >> 4) )
  {
LABEL_10:
    v14 = 0;
    goto LABEL_11;
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_17;
  v11 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
  if ( (a3[7] & 0x80u) == 0 )
    BUG();
  v12 = sub_BD2BC0((__int64)a3);
  v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
LABEL_11:
  *(_DWORD *)(a1 + 672) += qword_5030168 * ((32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF) - 32 - v7 - v14) >> 5);
  if ( a4 )
    return sub_30DFEC0(a1, a2, (__int64)a3);
  result = (unsigned int)qword_502FFA8;
  *(_DWORD *)(a1 + 660) += qword_502FFA8;
  return result;
}
