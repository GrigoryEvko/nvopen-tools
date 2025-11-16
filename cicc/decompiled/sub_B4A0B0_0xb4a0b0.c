// Function: sub_B4A0B0
// Address: 0xb4a0b0
//
__int64 __fastcall sub_B4A0B0(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int16 v13; // ax
  __int64 v14; // rdx
  char v15; // r15
  unsigned __int8 v16; // cl
  char v18; // [rsp+Fh] [rbp-41h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20[7]; // [rsp+18h] [rbp-38h] BYREF

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
LABEL_24:
      BUG();
LABEL_10:
    v9 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v5 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_10;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_24;
  v6 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v7 = sub_BD2BC0(a1);
  v9 = 32LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
LABEL_11:
  v10 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  v11 = (32 * v10 - 32 - v2 - v9) >> 5;
  if ( !(_DWORD)v11 )
    return 0;
  v12 = 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a1[32 * (v12 - v10)] + 8LL) + 8LL) == 14 )
    {
      v20[0] = *((_QWORD *)a1 + 9);
      v20[0] = sub_A744E0(v20, v12);
      v13 = sub_A73B10(v20);
      v14 = *((_QWORD *)a1 - 4);
      v15 = v13;
      v16 = HIBYTE(v13);
      if ( !*(_BYTE *)v14 )
      {
        v18 = HIBYTE(v13);
        v19 = *(_QWORD *)(v14 + 120);
        v20[0] = sub_A744E0(&v19, v12);
        v13 = sub_A73B10(v20);
        LOBYTE(v13) = v13 & v15;
        v16 = HIBYTE(v13) & v18;
      }
      if ( (v16 & (unsigned __int8)~(_BYTE)v13 & 0xF) != 0 )
        break;
    }
    if ( ++v12 == (unsigned int)v11 )
      return 0;
    v10 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  }
  return 1;
}
