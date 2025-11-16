// Function: sub_A7F480
// Address: 0xa7f480
//
__int64 __fastcall sub_A7F480(__int64 a1, unsigned __int8 *a2, int a3)
{
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v22; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v23; // [rsp+8h] [rbp-78h]
  int v24; // [rsp+Ch] [rbp-74h]
  _QWORD v25[2]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v26[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v27; // [rsp+40h] [rbp-40h]

  v5 = *((_DWORD *)a2 + 1);
  v6 = *((_QWORD *)a2 + 1);
  v24 = 0;
  v7 = v5 & 0x7FFFFFF;
  v22 = v6;
  v8 = *(_QWORD *)&a2[-32 * v7];
  v27 = 257;
  v9 = *(_QWORD *)&a2[32 * (1 - v7)];
  v25[0] = v8;
  v25[1] = v9;
  v10 = sub_B33D10(a1, a3, (unsigned int)&v22, 1, (unsigned int)v25, 2, v23, (__int64)v26);
  v11 = *a2;
  v12 = v10;
  if ( v11 == 40 )
  {
    v13 = 32LL * (unsigned int)sub_B491D0(a2);
  }
  else
  {
    v13 = 0;
    if ( v11 != 85 )
    {
      v13 = 64;
      if ( v11 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v14 = sub_BD2BC0(a2);
  v16 = v14 + v15;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v16 >> 4) )
LABEL_17:
      BUG();
LABEL_10:
    v20 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v16 - sub_BD2BC0(a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_17;
  v17 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v18 = sub_BD2BC0(a2);
  v20 = 32LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
LABEL_11:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v13 - v20) >> 5) == 4 )
    return sub_A7EE20(
             a1,
             *(_BYTE **)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
             v12,
             *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
  return v12;
}
