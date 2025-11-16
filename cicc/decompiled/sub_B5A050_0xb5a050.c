// Function: sub_B5A050
// Address: 0xb5a050
//
unsigned __int64 __fastcall sub_B5A050(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r12
  char v12; // al
  __int64 v13; // rdx
  int v14; // eax

  v1 = *a1;
  if ( v1 == 40 )
  {
    v2 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v2 = 0;
    if ( v1 != 85 )
    {
      v2 = 64;
      if ( v1 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_10;
  v3 = sub_BD2BC0(a1);
  v5 = v3 + v4;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v5 >> 4) )
LABEL_21:
      BUG();
LABEL_10:
    v9 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v5 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_10;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_21;
  v6 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v7 = sub_BD2BC0(a1);
  v9 = 32LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
LABEL_11:
  v10 = *((_QWORD *)a1 - 4);
  if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *((_QWORD *)a1 + 10) )
    BUG();
  v11 = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v2 - v9) >> 5;
  v12 = sub_B6B000(*(unsigned int *)(v10 + 36));
  v13 = *((_QWORD *)a1 - 4);
  v14 = (v12 == 0) + (_DWORD)v11 - 2;
  if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *((_QWORD *)a1 + 10) )
    BUG();
  return (__PAIR64__(v14, *(_DWORD *)(v13 + 36) - 103) - 2) >> 32;
}
