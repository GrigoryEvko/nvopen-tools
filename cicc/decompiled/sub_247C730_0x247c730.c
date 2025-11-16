// Function: sub_247C730
// Address: 0x247c730
//
__int64 __fastcall sub_247C730(__int64 *a1, unsigned __int8 *a2, int a3)
{
  __int64 v3; // r14
  int v4; // edx
  unsigned __int8 v5; // al
  unsigned __int8 v6; // cl
  int v7; // edx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  unsigned __int8 *v18; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // r12
  _BYTE *v22; // r13
  __int64 v23; // rax
  __int64 v24; // r9
  bool v25; // al
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rax
  _QWORD *v29; // rax
  unsigned __int64 v30; // rax
  _BYTE *v31; // [rsp+0h] [rbp-110h]
  __int64 v32; // [rsp+0h] [rbp-110h]
  __int64 v33; // [rsp+0h] [rbp-110h]
  __int64 v34; // [rsp+8h] [rbp-108h]
  __int64 v35; // [rsp+10h] [rbp-100h]
  __int64 v37; // [rsp+18h] [rbp-F8h]
  _QWORD v38[4]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v39; // [rsp+40h] [rbp-D0h]
  unsigned int *v40[24]; // [rsp+50h] [rbp-C0h] BYREF

  v3 = *((_QWORD *)a2 + 1);
  v4 = *(unsigned __int8 *)(v3 + 8);
  v5 = *(_BYTE *)(v3 + 8);
  if ( (unsigned int)(v4 - 17) > 1 )
  {
    if ( (_BYTE)v4 == 12 )
      goto LABEL_8;
  }
  else
  {
    v6 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
    if ( v6 == 12 )
      goto LABEL_8;
    if ( v4 == 18 )
      goto LABEL_4;
  }
  if ( v4 != 17 )
    goto LABEL_5;
  v6 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
LABEL_4:
  v5 = v6;
LABEL_5:
  if ( v5 > 3u && v5 != 5 && (v5 & 0xFD) != 4 )
    return 0;
LABEL_8:
  v7 = *a2;
  if ( v7 == 40 )
  {
    v8 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_19;
  }
  else
  {
    v8 = 0;
    if ( v7 != 85 )
    {
      v8 = 64;
      if ( v7 != 34 )
        BUG();
    }
    if ( (a2[7] & 0x80u) == 0 )
      goto LABEL_19;
  }
  v9 = sub_BD2BC0((__int64)a2);
  v11 = v9 + v10;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v11 >> 4) )
LABEL_51:
      BUG();
LABEL_19:
    v15 = 0;
    goto LABEL_20;
  }
  if ( !(unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_19;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_51;
  v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v13 = sub_BD2BC0((__int64)a2);
  v15 = 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
LABEL_20:
  v16 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v17 = (32 * v16 - 32 - v8 - v15) >> 5;
  if ( (_DWORD)v17 != a3 )
  {
    v18 = &a2[-32 * v16];
    while ( v3 == *(_QWORD *)(*(_QWORD *)v18 + 8LL) )
    {
      v18 += 32;
      if ( &a2[32 * ((unsigned int)(v17 - a3 - 1) - v16) + 32] == v18 )
        goto LABEL_30;
    }
    return 0;
  }
LABEL_30:
  sub_23D0AB0((__int64)v40, (__int64)a2, 0, 0, 0);
  if ( (_DWORD)v17 )
  {
    v37 = 0;
    v20 = (unsigned int)v17;
    v21 = 0;
    v34 = v20;
    v35 = 0;
    while ( 1 )
    {
      v27 = *(_QWORD *)&a2[32 * (v37 - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
      v24 = sub_246F3F0((__int64)a1, v27);
      if ( *(_DWORD *)(a1[1] + 4) )
        break;
      v22 = 0;
      if ( v21 )
        goto LABEL_33;
      v21 = v24;
LABEL_39:
      if ( ++v37 == v34 )
        goto LABEL_46;
    }
    v33 = v24;
    v28 = sub_246EE10((__int64)a1, v27);
    v24 = v33;
    v22 = (_BYTE *)v28;
    if ( v21 )
    {
LABEL_33:
      v31 = (_BYTE *)sub_2464970(a1, v40, v24, *(_QWORD *)(v21 + 8), 0);
      v38[0] = "_msprop";
      v39 = 259;
      v23 = sub_A82480(v40, (_BYTE *)v21, v31, (__int64)v38);
      v24 = (__int64)v31;
      v21 = v23;
    }
    else
    {
      v21 = v33;
    }
    if ( *(_DWORD *)(a1[1] + 4) )
    {
      if ( v35 )
      {
        if ( *v22 > 0x15u || (v32 = v24, v25 = sub_AC30F0((__int64)v22), v24 = v32, !v25) )
        {
          v39 = 257;
          v26 = sub_2465600((__int64)a1, v24, (__int64)v40, (__int64)v38);
          v39 = 257;
          v35 = sub_B36550(v40, v26, (__int64)v22, v35, (__int64)v38, 0);
        }
      }
      else
      {
        v35 = (__int64)v22;
      }
    }
    goto LABEL_39;
  }
  v35 = 0;
  v21 = 0;
LABEL_46:
  v29 = sub_2463540(a1, *((_QWORD *)a2 + 1));
  v30 = sub_2464970(a1, v40, v21, (__int64)v29, 0);
  sub_246EF60((__int64)a1, (__int64)a2, v30);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_246F1C0((__int64)a1, (__int64)a2, v35);
  sub_F94A20(v40, (__int64)a2);
  return 1;
}
