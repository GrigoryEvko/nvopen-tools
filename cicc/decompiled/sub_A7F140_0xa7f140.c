// Function: sub_A7F140
// Address: 0xa7f140
//
__int64 __fastcall sub_A7F140(unsigned int **a1, unsigned __int8 *a2, char a3, char a4)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rax
  int v14; // esi
  unsigned int v15; // r11d
  unsigned int v16; // r14d
  unsigned int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // rax
  int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // rcx
  __int64 v36; // [rsp+0h] [rbp-B0h]
  unsigned int v38; // [rsp+Ch] [rbp-A4h]
  __int64 v39; // [rsp+10h] [rbp-A0h]
  __int64 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-88h]
  int v43; // [rsp+2Ch] [rbp-84h]
  _QWORD v44[4]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE v45[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v46; // [rsp+70h] [rbp-40h]

  v6 = *((_QWORD *)a2 + 1);
  v7 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v41 = v6;
  v8 = *(_QWORD *)&a2[-32 * v7];
  v9 = *(_QWORD *)&a2[32 * (1 - v7)];
  v10 = *(_QWORD *)&a2[32 * (2 - v7)];
  v11 = *(_QWORD *)(v10 + 8);
  if ( a3 )
  {
    v12 = v6 == v11;
    v13 = v8;
    v14 = 181;
    v8 = v9;
    v9 = v13;
    if ( v12 )
      goto LABEL_6;
  }
  else
  {
    v14 = 180;
    if ( v6 == v11 )
      goto LABEL_6;
  }
  v15 = *(_DWORD *)(v6 + 32);
  v46 = 257;
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v36 = v8;
  v38 = v15;
  v39 = v10;
  v40 = v6;
  v16 = sub_BCB060(*(_QWORD *)(v10 + 8));
  v17 = sub_BCB060(v40);
  v18 = sub_A7EAA0(a1, (unsigned int)(v16 <= v17) + 38, v39, v40, (__int64)v45, 0, v44[0], 0);
  v46 = 257;
  v19 = sub_B37A60(a1, v38, v18, v45);
  v8 = v36;
  v10 = v19;
  v14 = 180 - ((a3 == 0) - 1);
LABEL_6:
  v44[2] = v10;
  v43 = 0;
  v46 = 257;
  v44[0] = v8;
  v44[1] = v9;
  v20 = sub_B33D10((_DWORD)a1, v14, (unsigned int)&v41, 1, (unsigned int)v44, 3, v42, (__int64)v45);
  v21 = *a2;
  v22 = v20;
  if ( v21 == 40 )
  {
    v23 = 32LL * (unsigned int)sub_B491D0(a2);
  }
  else
  {
    v23 = 0;
    if ( v21 != 85 )
    {
      v23 = 64;
      if ( v21 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_15;
  v24 = sub_BD2BC0(a2);
  v26 = v24 + v25;
  v27 = v24 + v25;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v27 >> 4) )
LABEL_29:
      BUG();
LABEL_15:
    v31 = 0;
    goto LABEL_16;
  }
  if ( !(unsigned int)((v26 - sub_BD2BC0(a2)) >> 4) )
    goto LABEL_15;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_29;
  v28 = *(_DWORD *)(sub_BD2BC0(a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v29 = sub_BD2BC0(a2);
  v31 = 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
LABEL_16:
  v32 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v33 = (32 * v32 - 32 - v23 - v31) >> 5;
  if ( (unsigned int)v33 > 3 )
  {
    if ( (_DWORD)v33 == 5 )
    {
      v34 = *(_QWORD *)&a2[32 * (3 - v32)];
    }
    else if ( a4 )
    {
      v34 = sub_AC9350(*((_QWORD *)a2 + 1));
      v32 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
    }
    else
    {
      v34 = *(_QWORD *)&a2[-32 * v32];
    }
    return sub_A7EE20((__int64)a1, *(_BYTE **)&a2[32 * ((unsigned int)(v33 - 1) - v32)], v22, v34);
  }
  return v22;
}
