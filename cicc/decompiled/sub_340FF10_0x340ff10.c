// Function: sub_340FF10
// Address: 0x340ff10
//
__int64 __fastcall sub_340FF10(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  unsigned __int16 v9; // cx
  __int64 v12; // r13
  __int64 v14; // rax
  unsigned __int16 v15; // bx
  __int64 v16; // rsi
  __int64 v18; // rax
  unsigned __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __int64 v27; // rdx
  char v28; // si
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int128 v34; // [rsp-30h] [rbp-E0h]
  __int128 v35; // [rsp-30h] [rbp-E0h]
  unsigned __int64 v36; // [rsp+8h] [rbp-A8h]
  unsigned __int16 v37; // [rsp+10h] [rbp-A0h]
  unsigned __int16 v38; // [rsp+14h] [rbp-9Ch]
  unsigned __int8 v39; // [rsp+14h] [rbp-9Ch]
  unsigned __int16 v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+20h] [rbp-90h] BYREF
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+30h] [rbp-80h]
  __int64 v44; // [rsp+38h] [rbp-78h]
  __int64 v45; // [rsp+40h] [rbp-70h]
  __int64 v46; // [rsp+48h] [rbp-68h]
  unsigned __int16 v47; // [rsp+50h] [rbp-60h] BYREF
  __int64 v48; // [rsp+58h] [rbp-58h]
  __int64 v49; // [rsp+60h] [rbp-50h] BYREF
  __int64 v50; // [rsp+68h] [rbp-48h]
  __int64 v51; // [rsp+70h] [rbp-40h]
  __int64 v52; // [rsp+78h] [rbp-38h]

  v9 = a3;
  v12 = a6;
  v14 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v41 = a3;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v42 = a4;
  if ( v15 == (_WORD)a3 )
  {
    if ( v15 )
      return a5;
    if ( v16 == v42 )
      goto LABEL_33;
    v50 = v16;
    LOWORD(v49) = 0;
LABEL_5:
    v38 = a3;
    v18 = sub_3007260((__int64)&v49);
    v9 = v38;
    v45 = v18;
    v19 = v18;
    v46 = v20;
    a6 = (unsigned __int8)v20;
    if ( !v38 )
      goto LABEL_6;
LABEL_18:
    if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      goto LABEL_35;
    v24 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
    if ( byte_444C4A0[16 * v9 - 8] )
      goto LABEL_21;
    goto LABEL_7;
  }
  LOWORD(v49) = v15;
  v50 = v16;
  if ( !v15 )
    goto LABEL_5;
  if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
    goto LABEL_35;
  v19 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
  a6 = (unsigned __int8)byte_444C4A0[16 * v15 - 8];
  if ( (_WORD)a3 )
    goto LABEL_18;
LABEL_6:
  v37 = v9;
  v36 = v19;
  v39 = a6;
  v21 = sub_3007260((__int64)&v41);
  v9 = v37;
  v19 = v36;
  v22 = v21;
  a6 = v39;
  v44 = v23;
  LOBYTE(v21) = v23;
  v24 = v22;
  v43 = v22;
  if ( (_BYTE)v21 )
    goto LABEL_21;
LABEL_7:
  if ( !(_BYTE)a6 )
  {
LABEL_21:
    if ( v19 < v24 )
    {
      *((_QWORD *)&v35 + 1) = v12;
      *(_QWORD *)&v35 = a5;
      return sub_340F900(a1, 0x1CBu, a2, v41, v42, a6, v35, a7, a8);
    }
  }
  if ( v15 != v9 )
    goto LABEL_9;
  if ( v15 )
    return a5;
LABEL_33:
  if ( v16 == v42 )
    return a5;
LABEL_9:
  v47 = v15;
  v48 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 504) <= 7u )
      goto LABEL_35;
    v33 = 16LL * (v15 - 1) + 71615648;
    v26 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
    v28 = *(_BYTE *)(v33 + 8);
    if ( !v9 )
      goto LABEL_11;
    goto LABEL_29;
  }
  v40 = v9;
  v25 = sub_3007260((__int64)&v47);
  v9 = v40;
  v51 = v25;
  v26 = v25;
  v52 = v27;
  v28 = v27;
  if ( v40 )
  {
LABEL_29:
    if ( v9 != 1 && (unsigned __int16)(v9 - 504) > 7u )
    {
      v32 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
      LOBYTE(v31) = byte_444C4A0[16 * v9 - 8];
      goto LABEL_12;
    }
LABEL_35:
    BUG();
  }
LABEL_11:
  v29 = sub_3007260((__int64)&v41);
  v31 = v30;
  v49 = v29;
  v32 = v29;
  v50 = v31;
LABEL_12:
  if ( (_BYTE)v31 && !v28 || v26 <= v32 )
    return a5;
  *((_QWORD *)&v34 + 1) = v12;
  *(_QWORD *)&v34 = a5;
  return sub_340F900(a1, 0x1CAu, a2, v41, v42, a6, v34, a7, a8);
}
