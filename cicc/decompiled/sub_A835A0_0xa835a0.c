// Function: sub_A835A0
// Address: 0xa835a0
//
__int64 __fastcall sub_A835A0(__int64 a1, unsigned __int8 *a2, int a3, char a4)
{
  __int64 v6; // rdi
  __int64 v7; // r9
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  int v12; // edx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v22; // esi
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE v25[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v26; // [rsp+20h] [rbp-40h]

  v6 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v7 = *(_QWORD *)&a2[-32 * v6];
  v8 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 32LL);
  if ( a3 == 3 )
  {
    v9 = sub_BCB2A0(*(_QWORD *)(a1 + 72));
    v10 = sub_BCDA70(v9, v8);
    v11 = sub_AD6530(v10);
  }
  else
  {
    if ( a3 != 7 )
    {
      switch ( a3 )
      {
        case 0:
          v22 = 32;
          goto LABEL_20;
        case 1:
          v22 = a4 == 0 ? 36 : 40;
          goto LABEL_20;
        case 2:
          v22 = a4 == 0 ? 37 : 41;
          goto LABEL_20;
        case 4:
          v22 = 33;
          goto LABEL_20;
        case 5:
          v22 = a4 == 0 ? 35 : 39;
          goto LABEL_20;
        case 6:
          v22 = a4 == 0 ? 34 : 38;
LABEL_20:
          v26 = 257;
          v11 = sub_92B530((unsigned int **)a1, v22, v7, *(_BYTE **)&a2[32 * (1 - v6)], (__int64)v25);
          goto LABEL_5;
        default:
          goto LABEL_29;
      }
    }
    v23 = sub_BCB2A0(*(_QWORD *)(a1 + 72));
    v24 = sub_BCDA70(v23, v8);
    v11 = sub_AD62B0(v24);
  }
LABEL_5:
  v12 = *a2;
  if ( v12 == 40 )
  {
    v13 = 32LL * (unsigned int)sub_B491D0(a2);
  }
  else
  {
    v13 = 0;
    if ( v12 != 85 )
    {
      v13 = 64;
      if ( v12 != 34 )
LABEL_29:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_18;
  v14 = sub_BD2BC0(a2);
  v16 = v14 + v15;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v16 >> 4) )
LABEL_27:
      BUG();
LABEL_18:
    v20 = 0;
    return sub_A80050(
             a1,
             v11,
             *(_BYTE **)&a2[32
                          * ((unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v13 - v20) >> 5)
                           - 1
                           - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
  }
  if ( !(unsigned int)((v16 - sub_BD2BC0(a2)) >> 4) )
    goto LABEL_18;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_27;
  v17 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v18 = sub_BD2BC0(a2);
  v20 = 32LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
  return sub_A80050(
           a1,
           v11,
           *(_BYTE **)&a2[32
                        * ((unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v13 - v20) >> 5)
                         - 1
                         - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
}
