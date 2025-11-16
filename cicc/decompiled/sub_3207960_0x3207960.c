// Function: sub_3207960
// Address: 0x3207960
//
__int64 __fastcall sub_3207960(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r15
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  char v6; // r14
  unsigned __int8 *v7; // rax
  int v8; // eax
  __int64 v9; // rdx
  int v10; // r12d
  unsigned __int8 v11; // al
  __int64 v12; // r15
  int v13; // r15d
  bool v14; // zf
  int v15; // eax
  int v16; // edx
  unsigned int v17; // eax
  int v18; // ecx
  __int16 v19; // ax
  __int64 v20; // rax
  unsigned int v22; // eax
  __int16 v24; // [rsp+20h] [rbp-50h] BYREF
  int v25; // [rsp+22h] [rbp-4Eh]
  int v26; // [rsp+28h] [rbp-48h]
  int v27; // [rsp+2Ch] [rbp-44h]
  __int16 v28; // [rsp+30h] [rbp-40h]
  char v29; // [rsp+32h] [rbp-3Eh]

  v3 = a2 - 16;
  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a2 - 32);
  else
    v5 = v3 - 8LL * ((v4 >> 2) & 0xF);
  v6 = **(_BYTE **)(v5 + 24);
  v7 = (unsigned __int8 *)sub_AF2CE0(a2);
  v8 = sub_3206530(a1, v7, 0);
  v9 = 0;
  v10 = v8;
  if ( v6 == 15 )
    v9 = sub_AF2CE0(a2);
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) != 0 )
    v12 = *(_QWORD *)(a2 - 32);
  else
    v12 = v3 - 8LL * ((v11 >> 2) & 0xF);
  v13 = sub_3206530(a1, *(unsigned __int8 **)(v12 + 24), v9);
  v14 = sub_AE2980(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0)[1] >> 3 == 8;
  v15 = *(_DWORD *)(a2 + 20);
  v16 = (unsigned __int8)(*(_QWORD *)(a2 + 24) >> 3);
  if ( v6 == 15 )
  {
    v22 = v15 & 0x30000;
    if ( v22 == 0x20000 )
    {
      v18 = 96;
      v19 = 6;
      goto LABEL_13;
    }
    if ( v22 > 0x20000 )
    {
      if ( v22 == 196608 )
      {
        v18 = 96;
        v19 = 7;
        goto LABEL_13;
      }
    }
    else
    {
      if ( !v22 )
      {
        v18 = 96;
        v19 = v16 != 0 ? 8 : 0;
        goto LABEL_13;
      }
      if ( v22 == 0x10000 )
      {
        v18 = 96;
        v19 = 5;
        goto LABEL_13;
      }
    }
    goto LABEL_29;
  }
  v17 = v15 & 0x30000;
  if ( v17 == 0x20000 )
  {
    v18 = 64;
    v19 = 2;
    goto LABEL_13;
  }
  if ( v17 > 0x20000 )
  {
    if ( v17 == 196608 )
    {
      v18 = 64;
      v19 = 3;
      goto LABEL_13;
    }
LABEL_29:
    BUG();
  }
  if ( !v17 )
  {
    v18 = 64;
    v19 = v16 != 0 ? 4 : 0;
    goto LABEL_13;
  }
  if ( v17 != 0x10000 )
    goto LABEL_29;
  v18 = 64;
  v19 = 1;
LABEL_13:
  v24 = 4098;
  v25 = v13;
  v27 = v10;
  v26 = v18 | (unsigned __int8)(2 * v14 + 10) | a3 | (v16 << 13);
  v28 = v19;
  v29 = 1;
  v20 = sub_3708FB0(a1 + 648, &v24);
  return sub_3707F80(a1 + 632, v20);
}
