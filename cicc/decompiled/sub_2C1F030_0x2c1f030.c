// Function: sub_2C1F030
// Address: 0x2c1f030
//
__int64 *__fastcall sub_2C1F030(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 *v5; // rdi
  char v6; // al
  char v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rax
  char v19; // [rsp+7h] [rbp-B9h]
  __int64 v20; // [rsp+8h] [rbp-B8h]
  __int64 v21; // [rsp+8h] [rbp-B8h]
  __int64 v22; // [rsp+10h] [rbp-B0h]
  __int64 v23; // [rsp+18h] [rbp-A8h]
  unsigned int **v24; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+28h] [rbp-98h]
  unsigned int **v26; // [rsp+30h] [rbp-90h] BYREF
  __int64 v27; // [rsp+38h] [rbp-88h]
  __int64 v28; // [rsp+40h] [rbp-80h]
  __int64 v29; // [rsp+48h] [rbp-78h]
  int v30; // [rsp+50h] [rbp-70h]
  char v31; // [rsp+54h] [rbp-6Ch]
  __int64 v32[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v33; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 96);
  if ( *(_BYTE *)v4 == 61 )
    v5 = *(__int64 **)(v4 + 8);
  else
    v5 = *(__int64 **)(*(_QWORD *)(v4 - 64) + 8LL);
  v20 = sub_BCE1B0(v5, *(_QWORD *)(a2 + 8));
  v6 = sub_2AAE0E0(*(_QWORD *)(a1 + 96));
  v7 = *(_BYTE *)(a1 + 104);
  v19 = v6;
  v23 = *(_QWORD *)(a2 + 904);
  v32[0] = *(_QWORD *)(a1 + 88);
  if ( v32[0] )
    sub_2AAAFA0(v32);
  sub_2BF1A90(a2, (__int64)v32);
  sub_9C6650(v32);
  v8 = *(_QWORD *)(a1 + 48);
  BYTE4(v32[0]) = 0;
  LODWORD(v32[0]) = 0;
  v22 = sub_2BFB120(a2, *(_QWORD *)(v8 + 8), (unsigned int *)v32);
  v9 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), v7);
  v10 = *(_BYTE *)(a1 + 106) == 0;
  v24 = (unsigned int **)v9;
  if ( v10 || (v11 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1))) == 0 )
  {
    v33 = 257;
    v14 = sub_ACD6D0(*(__int64 **)(v23 + 72));
    v12 = sub_B37620((unsigned int **)v23, *(_QWORD *)(a2 + 8), v14, v32);
LABEL_11:
    if ( v7 )
      goto LABEL_9;
    goto LABEL_12;
  }
  v12 = sub_2BFB640(a2, v11, 0);
  if ( !*(_BYTE *)(a1 + 105) )
    goto LABEL_11;
  v32[0] = (__int64)"vp.reverse.mask";
  v33 = 259;
  v12 = sub_2C0D550(v23, v12, v22, (__int64)v32);
  if ( v7 )
  {
LABEL_9:
    v28 = v12;
    v31 = 0;
    v26 = (unsigned int **)v23;
    LODWORD(v27) = 0;
    v29 = v22;
    v30 = 0;
    v32[0] = (__int64)"vp.op.load";
    v33 = 259;
    v13 = sub_10611B0(&v26, 32, v20, &v24, 1u, (__int64)v32);
    goto LABEL_13;
  }
LABEL_12:
  BYTE4(v25) = 0;
  v27 = v12;
  v32[0] = (__int64)"wide.masked.gather";
  v26 = v24;
  v33 = 259;
  v28 = v22;
  v13 = sub_B35180(v23, v20, 0x1B1u, (__int64)&v26, 3u, v25, (__int64)v32);
LABEL_13:
  v15 = (__int64 *)sub_BD5C60(v13);
  v16 = sub_A77A40(v15, v19);
  LODWORD(v32[0]) = 0;
  v21 = v16;
  v17 = (__int64 *)sub_BD5C60(v13);
  *(_QWORD *)(v13 + 72) = sub_A7B660((__int64 *)(v13 + 72), v17, v32, 1, v21);
  sub_2BF08A0(a2, (_BYTE *)v13, (_BYTE *)v4);
  if ( *(_BYTE *)(a1 + 105) )
  {
    v32[0] = (__int64)"vp.reverse";
    v33 = 259;
    v13 = sub_2C0D550(v23, v13, v22, (__int64)v32);
  }
  return sub_2BF26E0(a2, a1 + 112, v13, 0);
}
