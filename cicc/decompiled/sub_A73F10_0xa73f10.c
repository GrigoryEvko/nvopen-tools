// Function: sub_A73F10
// Address: 0xa73f10
//
__int64 __fastcall sub_A73F10(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r14
  __int64 v6; // r14
  __int64 v7; // r14
  __int64 v8; // r14
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // r14
  __int64 v14; // r14
  __int64 v15; // r14
  char v16; // bl
  __int16 v17; // ax
  unsigned __int8 v18; // r15
  __int16 v19; // ax
  char v20; // r14
  char v21; // al
  char v22; // cl
  __int64 v23; // r15
  char v24; // bl
  __int64 v25; // r15
  char v26; // bl
  __int64 v27; // r12
  __int16 v29; // kr00_2
  __int16 v30; // ax
  char v31; // r14
  __int16 v32; // ax
  unsigned __int8 v33; // r9
  char v34; // r10
  char v35; // di
  bool v36; // dl
  bool v37; // dl
  char v38; // [rsp+8h] [rbp-48h]
  unsigned __int8 v39; // [rsp+Eh] [rbp-42h]
  char v40; // [rsp+Fh] [rbp-41h]
  char v41; // [rsp+Fh] [rbp-41h]
  __int64 v42; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v43[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_B2D7D0(a2, 56);
  LOBYTE(v3) = v3 == sub_B2D7D0(a1, 56);
  v4 = sub_B2D7D0(a2, 63);
  LOBYTE(v3) = (v4 == sub_B2D7D0(a1, 63)) & v3;
  v5 = sub_B2D7D0(a2, 64);
  LOBYTE(v3) = (v5 == sub_B2D7D0(a1, 64)) & v3;
  v6 = sub_B2D7D0(a2, 59);
  LOBYTE(v3) = (v6 == sub_B2D7D0(a1, 59)) & v3;
  v7 = sub_B2D7D0(a2, 57);
  LOBYTE(v3) = (v7 == sub_B2D7D0(a1, 57)) & v3;
  v8 = sub_B2D7D0(a2, 58);
  LOBYTE(v3) = (v8 == sub_B2D7D0(a1, 58)) & v3;
  v9 = sub_B2D7D0(a2, 60);
  LOBYTE(v3) = (v9 == sub_B2D7D0(a1, 60)) & v3;
  v10 = sub_B2D7D0(a2, 61);
  LOBYTE(v3) = (v10 == sub_B2D7D0(a1, 61)) & v3;
  v11 = sub_B2D7D0(a2, 62);
  LOBYTE(v3) = (v11 == sub_B2D7D0(a1, 62)) & v3;
  v12 = sub_B2D7D0(a2, 55);
  LOBYTE(v3) = (v12 == sub_B2D7D0(a1, 55)) & v3;
  v13 = sub_B2D7D0(a2, 65);
  LOBYTE(v3) = (v13 == sub_B2D7D0(a1, 65)) & v3;
  v14 = sub_B2D7E0(a2, "use-sample-profile", 18);
  LOBYTE(v3) = (v14 == sub_B2D7E0(a1, "use-sample-profile", 18)) & v3;
  v15 = sub_B2D7D0(a2, 33);
  v16 = (v15 == sub_B2D7D0(a1, 33)) & v3;
  v17 = sub_B2D9D0(a1);
  v18 = v17;
  v40 = HIBYTE(v17);
  v19 = sub_B2D9D0(a2);
  v20 = (_BYTE)v19 == v18 && v40 == HIBYTE(v19);
  if ( !v20 )
  {
    if ( (_BYTE)v19 == 3 )
    {
      if ( HIBYTE(v19) != 3 && v40 != HIBYTE(v19) )
        goto LABEL_4;
    }
    else if ( (_BYTE)v19 != v18 || HIBYTE(v19) != 3 )
    {
      goto LABEL_4;
    }
  }
  v29 = v19;
  v30 = sub_B2DAA0(a1);
  v39 = v30;
  v38 = v30;
  v31 = HIBYTE(v30);
  v32 = sub_B2DAA0(a2);
  v33 = v39;
  v34 = v32;
  v35 = HIBYTE(v32);
  if ( v38 != -1 )
  {
LABEL_9:
    if ( (_BYTE)v32 != 0xFF )
    {
LABEL_10:
      v36 = v31 == v35;
      v20 = v31 == v35 && v34 == (char)v33;
      if ( v20 )
        goto LABEL_4;
      goto LABEL_11;
    }
LABEL_22:
    if ( HIBYTE(v32) != 0xFF )
    {
      v37 = v33 == 0xFF;
      v20 = v37 && v31 == HIBYTE(v32);
      if ( !v20 )
      {
        v20 = HIBYTE(v32) == 3;
        goto LABEL_13;
      }
      goto LABEL_4;
    }
    v35 = HIBYTE(v29);
    v34 = v29;
    goto LABEL_10;
  }
  if ( v31 == -1 )
  {
    v31 = v40;
    v33 = v18;
    goto LABEL_9;
  }
  if ( (_BYTE)v32 == 0xFF )
    goto LABEL_22;
  v36 = v31 == HIBYTE(v32);
LABEL_11:
  v20 = v35 == 3;
  if ( v34 != 3 )
  {
    v37 = v34 == (char)v33;
LABEL_13:
    v20 &= v37;
    goto LABEL_4;
  }
  v20 |= v36;
  if ( !v20 )
  {
    v37 = v33 == 3;
    goto LABEL_13;
  }
LABEL_4:
  v42 = *(_QWORD *)(a2 + 120);
  v21 = sub_A73ED0(&v42, 72);
  v22 = 1;
  if ( v21 )
  {
    v43[0] = *(_QWORD *)(a1 + 120);
    v22 = sub_A73ED0(v43, 72);
  }
  v41 = v22;
  v23 = sub_B2D7E0(a2, "sign-return-address", 19);
  v24 = (v23 == sub_B2D7E0(a1, "sign-return-address", 19)) & v16;
  v25 = sub_B2D7E0(a2, "sign-return-address-key", 23);
  v26 = (v25 == sub_B2D7E0(a1, "sign-return-address-key", 23)) & v24;
  v27 = sub_B2D7E0(a2, "branch-protection-pauth-lr", 26);
  return (unsigned __int8)(v26 & (v27 == sub_B2D7E0(a1, "branch-protection-pauth-lr", 26)) & v20) & (unsigned __int8)v41;
}
