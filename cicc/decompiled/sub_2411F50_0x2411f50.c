// Function: sub_2411F50
// Address: 0x2411f50
//
__int64 __fastcall sub_2411F50(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int *v15; // rdx
  int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 **v19; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rdi
  unsigned __int8 v28; // r13
  int v29; // r14d
  _QWORD *v30; // rax
  int v31; // ecx
  __int64 v32; // rsi
  _QWORD *v33; // [rsp+8h] [rbp-108h]
  char v34; // [rsp+10h] [rbp-100h]
  __int64 v35; // [rsp+18h] [rbp-F8h]
  __int64 v36; // [rsp+18h] [rbp-F8h]
  _BYTE v37[32]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v38; // [rsp+40h] [rbp-D0h]
  unsigned int *v39[24]; // [rsp+50h] [rbp-C0h] BYREF

  v4 = sub_B2BEC0(a1[1]);
  v7 = a1[1];
  v8 = v4;
  if ( (*(_BYTE *)(v7 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1[1], a2, v5, v6);
    v9 = *(_QWORD *)(v7 + 96);
    v35 = v9 + 40LL * *(_QWORD *)(v7 + 104);
    if ( (*(_BYTE *)(v7 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v7, a2, v21, v22);
      v9 = *(_QWORD *)(v7 + 96);
    }
  }
  else
  {
    v9 = *(_QWORD *)(v7 + 96);
    v35 = v9 + 40LL * *(_QWORD *)(v7 + 104);
  }
  v10 = 0;
  if ( v9 == v35 )
    goto LABEL_8;
  while ( 1 )
  {
    v11 = *(_QWORD *)(v9 + 8);
    v12 = *(unsigned __int8 *)(v11 + 8);
    if ( (unsigned __int8)v12 > 0xCu || (v13 = 4143, !_bittest64(&v13, v12)) )
    {
      if ( (v12 & 0xFD) != 4 && (v12 & 0xFB) != 0xA )
      {
        if ( (unsigned __int8)(v12 - 15) > 3u && (_BYTE)v12 != 20 || !(unsigned __int8)sub_BCEBA0(v11, 0) )
        {
          if ( a2 == v9 )
            goto LABEL_8;
          goto LABEL_16;
        }
        v11 = *(_QWORD *)(v9 + 8);
      }
    }
    v33 = sub_240F000(*a1, v11);
    v34 = sub_AE5020(v8, (__int64)v33);
    v14 = sub_9208B0(v8, (__int64)v33);
    v39[1] = v15;
    v39[0] = (unsigned int *)((((unsigned __int64)(v14 + 7) >> 3) + (1LL << v34) - 1) >> v34 << v34);
    v16 = sub_CA1930(v39);
    if ( a2 == v9 )
      break;
    v10 += ((1 << byte_4FE3AA9) + v16 - 1) & -(1 << byte_4FE3AA9);
    if ( v10 > 0x320 )
      goto LABEL_8;
LABEL_16:
    v9 += 40;
    if ( v9 == v35 )
      goto LABEL_8;
  }
  if ( v10 + v16 > 0x320 )
  {
LABEL_8:
    v17 = *(_QWORD *)(a2 + 8);
    v18 = *a1;
    if ( (unsigned __int8)(*(_BYTE *)(v17 + 8) - 15) > 1u )
      return *(_QWORD *)(v18 + 72);
    v19 = (__int64 **)sub_240F000(v18, v17);
    return sub_AC9350(v19);
  }
  v23 = *(_QWORD *)(a1[1] + 80);
  if ( !v23 )
    BUG();
  v24 = *(_QWORD *)(v23 + 32);
  if ( v24 )
    v24 -= 24;
  sub_23D0AB0((__int64)v39, v24, 0, 0, 0);
  v25 = sub_24105D0(a1, v10, (__int64)v39);
  v26 = *(_QWORD *)(a2 + 8);
  v27 = *a1;
  v28 = byte_4FE3AA9;
  v29 = v25;
  v38 = 257;
  v30 = sub_240F000(v27, v26);
  v31 = v28;
  v32 = (__int64)v30;
  BYTE1(v31) = 1;
  v36 = sub_A82CA0(v39, (__int64)v30, v29, v31, 0, (__int64)v37);
  sub_F94A20(v39, v32);
  return v36;
}
