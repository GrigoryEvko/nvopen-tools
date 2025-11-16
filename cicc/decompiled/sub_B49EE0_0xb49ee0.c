// Function: sub_B49EE0
// Address: 0xb49ee0
//
__int16 __fastcall sub_B49EE0(unsigned __int8 *a1, unsigned int a2)
{
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int16 result; // ax
  __int16 v12; // bx
  char v13; // dl
  char v14; // r15
  __int64 v15; // rax
  __int16 v16; // ax
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *a1;
  if ( v2 == 40 )
  {
    v3 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v3 = 0;
    if ( v2 != 85 )
    {
      v3 = 64;
      if ( v2 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_10;
  v4 = sub_BD2BC0(a1);
  v6 = v4 + v5;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v6 >> 4) )
LABEL_22:
      BUG();
LABEL_10:
    v10 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v6 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_10;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_22;
  v7 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v8 = sub_BD2BC0(a1);
  v10 = 32LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
LABEL_11:
  if ( a2 >= (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v3 - v10) >> 5) )
  {
    if ( !*(_DWORD *)(*(_QWORD *)sub_B49810((__int64)a1, a2) + 8LL) )
      return 0;
    return 3855;
  }
  else
  {
    if ( (unsigned __int8)sub_B49B80((__int64)a1, a2, 81) )
      return 0;
    v18[0] = *((_QWORD *)a1 + 9);
    v18[0] = sub_A744E0(v18, a2);
    v12 = sub_A73B10(v18);
    v13 = v12;
    v14 = HIBYTE(v12);
    v15 = *((_QWORD *)a1 - 4);
    if ( !*(_BYTE *)v15 )
    {
      v17 = *(_QWORD *)(v15 + 120);
      v18[0] = sub_A744E0(&v17, a2);
      v16 = sub_A73B10(v18);
      v13 = v16 & v12;
      v14 = HIBYTE(v16) & HIBYTE(v12);
    }
    LOBYTE(result) = v13;
    HIBYTE(result) = v14;
  }
  return result;
}
