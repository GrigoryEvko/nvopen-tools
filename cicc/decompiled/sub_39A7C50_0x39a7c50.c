// Function: sub_39A7C50
// Address: 0x39a7c50
//
__int64 __fastcall sub_39A7C50(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdi
  void *v9; // rax
  size_t v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r15
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r9
  __int64 *v15; // r15
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  int v18; // eax
  int v19; // edx
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // rax
  __int64 v23; // r8
  int v24; // r15d
  __int64 v25; // rax
  __int64 *v26; // r14
  __int64 v27; // rcx
  __int16 v28; // dx
  __int64 v29; // rax
  unsigned __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  unsigned __int64 v32; // [rsp+18h] [rbp-58h]
  unsigned __int64 v33; // [rsp+20h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-50h]
  unsigned __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38[8]; // [rsp+30h] [rbp-40h] BYREF

  v5 = sub_39A5A90((__int64)a1, *(_WORD *)(a3 + 2), a2, 0);
  v6 = *(unsigned int *)(a3 + 8);
  v7 = v5;
  v8 = *(_QWORD *)(a3 + 8 * (2 - v6));
  if ( v8 )
  {
    v9 = (void *)sub_161E970(v8);
    if ( v10 )
      sub_39A3F30(a1, v7, 3, v9, v10);
    v6 = *(unsigned int *)(a3 + 8);
  }
  v11 = *(_QWORD *)(a3 + 8 * (3 - v6));
  if ( v11 )
    sub_39A6760(a1, v7, v11, 73);
  sub_39A3790((__int64)a1, v7, a3);
  if ( *(_WORD *)(a3 + 2) == 28 && (*(_BYTE *)(a3 + 28) & 0x20) != 0 )
  {
    v29 = sub_145CBF0(a1 + 11, 16, 16);
    *(_QWORD *)v29 = 0;
    v26 = (__int64 *)v29;
    *(_DWORD *)(v29 + 8) = 0;
    sub_39A35E0((__int64)a1, (__int64 *)v29, 11, 18);
    sub_39A35E0((__int64)a1, v26, 11, 6);
    sub_39A35E0((__int64)a1, v26, 11, 16);
    sub_39A35E0((__int64)a1, v26, 15, *(_QWORD *)(a3 + 40));
    sub_39A35E0((__int64)a1, v26, 11, 28);
    sub_39A35E0((__int64)a1, v26, 11, 6);
    v27 = 34;
    v28 = 11;
    goto LABEL_39;
  }
  v12 = *(_QWORD *)(a3 + 32);
  v36 = v12;
  v13 = sub_397FBC0(a3);
  v14 = v13;
  if ( !v13 || v13 == v12 )
  {
    v24 = *(_DWORD *)(a3 + 48) >> 3;
    v37 = *(_QWORD *)(a3 + 40) >> 3;
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 4u && v24 )
    {
      LODWORD(v38[0]) = 65551;
      sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 136, (__int64)v38, v24 & 0x1FFFFFFF);
    }
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 2u )
    {
      v15 = (__int64 *)(v7 + 8);
      goto LABEL_35;
    }
    goto LABEL_38;
  }
  v15 = (__int64 *)(v7 + 8);
  if ( *(_BYTE *)(a1[25] + 4497) )
  {
    v35 = v13;
    BYTE2(v38[0]) = 0;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 11, (__int64)v38, v13 >> 3);
    v14 = v35;
  }
  v33 = v14;
  BYTE2(v38[0]) = 0;
  sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 13, (__int64)v38, v36);
  v16 = *(_QWORD *)(a3 + 40);
  v17 = (unsigned int)-(int)v33;
  if ( *(_BYTE *)(a1[25] + 4497) )
  {
    v32 = v33;
    v30 = v33 + v16;
    v31 = (v33 + v16) & v17;
    v34 = v31 - v33;
    if ( *(_BYTE *)sub_396DDB0(a1[24]) )
      v23 = v30 - v31;
    else
      v23 = v32 - v36 - v30 + v31;
    LODWORD(v38[0]) = 65551;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 12, (__int64)v38, v23);
    v37 = v34 >> 3;
  }
  else
  {
    BYTE2(v38[0]) = 0;
    v37 = (v16 & v17) >> 3;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 107, (__int64)v38, v16);
  }
  if ( (unsigned __int16)sub_398C0A0(a1[25]) <= 2u )
  {
LABEL_38:
    v25 = sub_145CBF0(a1 + 11, 16, 16);
    *(_QWORD *)v25 = 0;
    v26 = (__int64 *)v25;
    *(_DWORD *)(v25 + 8) = 0;
    sub_39A35E0((__int64)a1, (__int64 *)v25, 11, 35);
    v27 = v37;
    v28 = 15;
LABEL_39:
    sub_39A35E0((__int64)a1, v26, v28, v27);
    sub_39A4520(a1, v7, 56, (__int64 **)v26);
LABEL_17:
    v18 = *(_DWORD *)(a3 + 28);
    v19 = v18 & 3;
    if ( v19 != 2 )
      goto LABEL_18;
LABEL_36:
    LODWORD(v38[0]) = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 50, (__int64)v38, 2);
    if ( (*(_DWORD *)(a3 + 28) & 0x20) == 0 )
      goto LABEL_22;
LABEL_37:
    LODWORD(v38[0]) = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 76, (__int64)v38, 1);
    goto LABEL_22;
  }
  if ( !*(_BYTE *)(a1[25] + 4497) )
    goto LABEL_17;
LABEL_35:
  BYTE2(v38[0]) = 0;
  sub_39A3560((__int64)a1, v15, 56, (__int64)v38, v37);
  v18 = *(_DWORD *)(a3 + 28);
  v19 = v18 & 3;
  if ( v19 == 2 )
    goto LABEL_36;
LABEL_18:
  if ( v19 == 1 )
  {
    LODWORD(v38[0]) = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 50, (__int64)v38, 3);
    v18 = *(_DWORD *)(a3 + 28);
  }
  else if ( v19 == 3 )
  {
    LODWORD(v38[0]) = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(v7 + 8), 50, (__int64)v38, 1);
    v18 = *(_DWORD *)(a3 + 28);
  }
  if ( (v18 & 0x20) != 0 )
    goto LABEL_37;
LABEL_22:
  v20 = *(unsigned __int8 **)(a3 + 8 * (4LL - *(unsigned int *)(a3 + 8)));
  if ( v20 )
  {
    if ( *v20 == 27 )
    {
      v21 = sub_39A23D0((__int64)a1, v20);
      if ( v21 )
      {
        v38[1] = (__int64)v21;
        v38[0] = 0x133FED00000006LL;
        sub_39A31C0((__int64 *)(v7 + 8), a1 + 11, v38);
      }
    }
  }
  if ( (*(_BYTE *)(a3 + 28) & 0x40) != 0 )
    sub_39A34D0((__int64)a1, v7, 52);
  return v7;
}
