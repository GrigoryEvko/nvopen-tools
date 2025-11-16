// Function: sub_A7EEB0
// Address: 0xa7eeb0
//
__int64 __fastcall sub_A7EEB0(unsigned int **a1, unsigned __int8 *a2, char a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // r10
  __int64 v7; // rdx
  unsigned int v8; // r11d
  unsigned int v9; // r15d
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r15
  int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned int v25; // [rsp+Ch] [rbp-A4h]
  __int64 v26; // [rsp+10h] [rbp-A0h]
  __int64 v27; // [rsp+18h] [rbp-98h]
  __int64 v28; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-88h]
  int v30; // [rsp+2Ch] [rbp-84h]
  _QWORD v31[4]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE v32[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v33; // [rsp+70h] [rbp-40h]

  v4 = *((_QWORD *)a2 + 1);
  v5 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v28 = v4;
  v6 = *(_QWORD *)&a2[-32 * v5];
  v7 = *(_QWORD *)&a2[32 * (1 - v5)];
  if ( *(_QWORD *)(v7 + 8) != v4 )
  {
    v8 = *(_DWORD *)(v4 + 32);
    v33 = 257;
    if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
      v4 = **(_QWORD **)(v4 + 16);
    v26 = v6;
    v25 = v8;
    v27 = v7;
    v9 = sub_BCB060(*(_QWORD *)(v7 + 8));
    v10 = sub_BCB060(v4);
    v11 = sub_A7EAA0(a1, (unsigned int)(v9 <= v10) + 38, v27, v4, (__int64)v32, 0, v31[0], 0);
    v33 = 257;
    v12 = sub_B37A60(a1, v25, v11, v32);
    v6 = v26;
    v7 = v12;
  }
  v31[2] = v7;
  v33 = 257;
  v30 = 0;
  v31[0] = v6;
  v31[1] = v6;
  v13 = sub_B33D10(
          (_DWORD)a1,
          180 - ((unsigned int)(a3 == 0) - 1),
          (unsigned int)&v28,
          1,
          (unsigned int)v31,
          3,
          v29,
          (__int64)v32);
  v14 = *a2;
  v15 = v13;
  if ( v14 == 40 )
  {
    v16 = 32LL * (unsigned int)sub_B491D0(a2);
  }
  else
  {
    v16 = 0;
    if ( v14 != 85 )
    {
      v16 = 64;
      if ( v14 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_14;
  v17 = sub_BD2BC0(a2);
  v19 = v17 + v18;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v19 >> 4) )
LABEL_21:
      BUG();
LABEL_14:
    v23 = 0;
    goto LABEL_15;
  }
  if ( !(unsigned int)((v19 - sub_BD2BC0(a2)) >> 4) )
    goto LABEL_14;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_21;
  v20 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v21 = sub_BD2BC0(a2);
  v23 = 32LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20);
LABEL_15:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v16 - v23) >> 5) == 4 )
    return sub_A7EE20(
             (__int64)a1,
             *(_BYTE **)&a2[32 * (3LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))],
             v15,
             *(_QWORD *)&a2[32 * (2LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))]);
  return v15;
}
