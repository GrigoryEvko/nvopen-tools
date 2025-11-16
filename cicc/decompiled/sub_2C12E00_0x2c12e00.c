// Function: sub_2C12E00
// Address: 0x2c12e00
//
void __fastcall sub_2C12E00(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  char v4; // al
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 *v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-C8h]
  char v24; // [rsp+17h] [rbp-B9h]
  _BYTE *v25; // [rsp+18h] [rbp-B8h]
  unsigned int **v26; // [rsp+20h] [rbp-B0h]
  char v27; // [rsp+28h] [rbp-A8h]
  _QWORD v28[2]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int **v29; // [rsp+40h] [rbp-90h] BYREF
  __int64 v30; // [rsp+48h] [rbp-88h]
  __int64 v31; // [rsp+50h] [rbp-80h]
  __int64 v32; // [rsp+58h] [rbp-78h]
  int v33; // [rsp+60h] [rbp-70h]
  char v34; // [rsp+64h] [rbp-6Ch]
  __int64 v35[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v36; // [rsp+90h] [rbp-40h]

  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
  v25 = *(_BYTE **)(a1 + 96);
  v27 = *(_BYTE *)(a1 + 104);
  v4 = sub_2AAE0E0((__int64)v25);
  v5 = *(_QWORD *)(a2 + 904);
  v24 = v4;
  v35[0] = *(_QWORD *)(a1 + 88);
  if ( v35[0] )
    sub_2AAAFA0(v35);
  sub_2BF1A90(a2, (__int64)v35);
  sub_9C6650(v35);
  v6 = sub_2BFB640(a2, v3, 0);
  BYTE4(v35[0]) = 0;
  v26 = (unsigned int **)v6;
  v7 = *(_QWORD *)(a1 + 48);
  LODWORD(v35[0]) = 0;
  v8 = sub_2BFB120(a2, *(_QWORD *)(v7 + 16), (unsigned int *)v35);
  v9 = v8;
  if ( *(_BYTE *)(a1 + 105) )
  {
    v35[0] = (__int64)"vp.reverse";
    v36 = 259;
    v26 = (unsigned int **)sub_2C0D550(v5, (__int64)v26, v8, (__int64)v35);
  }
  if ( *(_BYTE *)(a1 + 106)
    && (v10 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 56) - 1))) != 0 )
  {
    v11 = sub_2BFB640(a2, v10, 0);
    if ( *(_BYTE *)(a1 + 105) )
    {
      v35[0] = (__int64)"vp.reverse.mask";
      v36 = 259;
      v11 = sub_2C0D550(v5, v11, v9, (__int64)v35);
    }
  }
  else
  {
    v36 = 257;
    v12 = sub_ACD6D0(*(__int64 **)(v5 + 72));
    v11 = sub_B37620((unsigned int **)v5, *(_QWORD *)(a2 + 8), v12, v35);
  }
  v23 = v11;
  v13 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), v27);
  v31 = v23;
  v36 = 257;
  if ( v27 )
  {
    v28[0] = v26;
    v29 = (unsigned int **)v5;
    v32 = v9;
    LODWORD(v30) = 0;
    v33 = 0;
    v34 = 0;
    v28[1] = v13;
    v21 = (_QWORD *)sub_BD5C60(v9);
    v22 = sub_BCB120(v21);
    v16 = sub_10611B0(&v29, 33, v22, v28, 2u, (__int64)v35);
  }
  else
  {
    v32 = v9;
    v29 = v26;
    HIDWORD(v28[0]) = 0;
    v30 = v13;
    v14 = (_QWORD *)sub_BD5C60(v9);
    v15 = sub_BCB120(v14);
    v16 = sub_B35180(v5, v15, 0x1D6u, (__int64)&v29, 4u, LODWORD(v28[0]), (__int64)v35);
  }
  v17 = (__int64 *)sub_BD5C60(v16);
  v18 = sub_A77A40(v17, v24);
  LODWORD(v35[0]) = 1;
  v19 = v18;
  v20 = (__int64 *)sub_BD5C60(v16);
  *(_QWORD *)(v16 + 72) = sub_A7B660((__int64 *)(v16 + 72), v20, v35, 1, v19);
  sub_2BF08A0(a2, (_BYTE *)v16, v25);
}
