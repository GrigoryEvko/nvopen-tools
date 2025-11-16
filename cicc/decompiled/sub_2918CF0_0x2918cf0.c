// Function: sub_2918CF0
// Address: 0x2918cf0
//
char __fastcall sub_2918CF0(_QWORD *a1, unsigned __int8 *a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rax
  int v4; // eax
  int v5; // edx
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // eax
  int v9; // edx
  unsigned int v10; // r12d
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx

  v2 = a1[42];
  v3 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  if ( v2 < (unsigned __int64)&a2[-v3] )
    goto LABEL_2;
  v4 = *a2;
  v5 = v4 - 29;
  if ( v4 == 40 )
  {
    v3 = (unsigned __int64)&a2[-32 - 32LL * (unsigned int)sub_B491D0((__int64)a2)];
    if ( v2 >= v3 )
      goto LABEL_2;
LABEL_6:
    v7 = sub_BD2910(a1[42]);
    LOWORD(v3) = sub_B49EE0(a2, v7);
    if ( (_WORD)v3 )
      goto LABEL_2;
    v8 = sub_BD2910(a1[42]);
    v9 = *a2;
    v10 = v8;
    if ( v9 == 40 )
    {
      v11 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    }
    else
    {
      v11 = 0;
      if ( v9 != 85 )
      {
        v11 = 64;
        if ( v9 != 34 )
          goto LABEL_31;
      }
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v12 = sub_BD2BC0((__int64)a2);
      v14 = v12 + v13;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v14 >> 4) )
          goto LABEL_33;
      }
      else if ( (unsigned int)((v14 - sub_BD2BC0((__int64)a2)) >> 4) )
      {
        if ( (a2[7] & 0x80u) != 0 )
        {
          v15 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v16 = sub_BD2BC0((__int64)a2);
          v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
LABEL_24:
          if ( v10 < (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v11 - v18) >> 5)
            && (LOBYTE(v3) = sub_B49B80((__int64)a2, v10, 81), (_BYTE)v3)
            || (LOBYTE(v3) = sub_CF49B0(a2, v10, 51), (_BYTE)v3)
            || (LOBYTE(v3) = sub_CF49B0(a2, v10, 50), (_BYTE)v3) )
          {
            a1[3] = a2;
            return v3;
          }
          goto LABEL_2;
        }
LABEL_33:
        BUG();
      }
    }
    v18 = 0;
    goto LABEL_24;
  }
  v6 = -32;
  if ( v5 != 56 )
  {
    v6 = -96;
    if ( v5 != 5 )
LABEL_31:
      BUG();
  }
  v3 = (unsigned __int64)&a2[v6];
  if ( v2 < v3 )
    goto LABEL_6;
LABEL_2:
  a1[2] = a2;
  a1[1] = a2;
  return v3;
}
