// Function: sub_30E0570
// Address: 0x30e0570
//
unsigned __int64 __fastcall sub_30E0570(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4)
{
  int v6; // edx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  int v11; // r14d
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // r12
  unsigned int v18; // r14d
  __int64 v19; // rax
  unsigned int v20; // eax

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
LABEL_22:
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
      goto LABEL_22;
    goto LABEL_10;
  }
  if ( !(unsigned int)((v10 - sub_BD2BC0((__int64)a3)) >> 4) )
  {
LABEL_10:
    v14 = 0;
    goto LABEL_11;
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_22;
  v11 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
  if ( (a3[7] & 0x80u) == 0 )
    BUG();
  v12 = sub_BD2BC0((__int64)a3);
  v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
LABEL_11:
  v15 = (unsigned int)qword_5030168 * (unsigned int)((32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF) - 32 - v7 - v14) >> 5);
  if ( v15 >= 0x80000000 )
    v15 = 0x7FFFFFFF;
  v16 = *(int *)(a1 + 716) + v15;
  if ( v16 >= 0x80000000LL )
    LODWORD(v16) = 0x7FFFFFFF;
  *(_DWORD *)(a1 + 716) = v16;
  if ( a4 && *(_BYTE *)(a1 + 712) )
    return sub_30E03A0(a1, a2, (__int64)a3);
  v17 = *(__int64 **)(a1 + 8);
  v18 = qword_502FFA8;
  v19 = sub_B491C0(*(_QWORD *)(a1 + 96));
  v20 = sub_DFE000(v17, v19, (__int64)a3, v18);
  return sub_30D0F50(a1, v20);
}
