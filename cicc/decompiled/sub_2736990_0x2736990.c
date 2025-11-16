// Function: sub_2736990
// Address: 0x2736990
//
__int64 __fastcall sub_2736990(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  __int64 v5; // rdx
  __int64 v6; // r8
  unsigned __int16 v7; // ax
  _QWORD *v8; // rdi
  __int64 v9; // r14
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // r11
  unsigned __int16 v14; // r9
  __int64 v15; // r8
  __int64 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // r15
  __int64 v20; // rsi
  unsigned int v21; // esi
  __int64 v22; // rdx
  unsigned __int8 *v23; // r8
  __int64 result; // rax
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r13
  unsigned __int8 *v28; // r13
  unsigned __int8 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdi
  __int64 *v33; // r15
  __int64 v34; // rsi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  int v41; // eax
  char v42; // al
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // r15
  __int64 v52; // rsi
  __int64 v53; // r12
  __int64 v54; // rsi
  unsigned __int8 *v55; // rsi
  __int64 v56; // [rsp+8h] [rbp-88h]
  unsigned __int16 v57; // [rsp+8h] [rbp-88h]
  unsigned __int16 v58; // [rsp+10h] [rbp-80h]
  __int64 v59; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v62; // [rsp+28h] [rbp-68h] BYREF
  __int64 v63[4]; // [rsp+30h] [rbp-60h] BYREF
  char v64; // [rsp+50h] [rbp-40h]
  char v65; // [rsp+51h] [rbp-3Fh]

  v3 = (_QWORD *)a2;
  v5 = *(_QWORD *)a3;
  if ( !v5 )
  {
    v37 = *(_QWORD *)(a3 + 8);
    if ( !v37 )
      goto LABEL_14;
    if ( *(_QWORD *)(a2 + 8) == v37 )
      goto LABEL_14;
    v38 = sub_BCB2D0(*(_QWORD **)(a1 + 24));
    v39 = sub_ACD640(v38, 0, 0);
    *(_QWORD *)a3 = v39;
    v5 = v39;
    if ( !v39 )
      goto LABEL_14;
  }
  v6 = *(_QWORD *)(a3 + 16);
  if ( *(_QWORD *)(a3 + 8) )
  {
    v7 = *(_WORD *)(a3 + 24);
    v56 = *(_QWORD *)(a3 + 16);
    v61 = v5;
    v58 = v7;
    v63[0] = (__int64)"mat_gep";
    v65 = 1;
    v8 = *(_QWORD **)(a1 + 24);
    v64 = 3;
    v9 = sub_BCB2B0(v8);
    v10 = sub_BD2C40(88, 2u);
    v11 = v56;
    v12 = (__int64)v10;
    if ( !v10 )
      goto LABEL_6;
    v13 = *(_QWORD *)(a2 + 8);
    v14 = v58;
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
    {
LABEL_5:
      sub_B44260(v12, v13, 34, 2u, v11, v14);
      *(_QWORD *)(v12 + 72) = v9;
      *(_QWORD *)(v12 + 80) = sub_B4DC50(v9, (__int64)&v61, 1);
      sub_B4D9A0(v12, a2, &v61, 1, (__int64)v63);
LABEL_6:
      v15 = *(_QWORD *)(a3 + 16);
      v65 = 1;
      v63[0] = (__int64)"mat_bitcast";
      v64 = 3;
      if ( v15 )
        v15 -= 24;
      v16 = v15 + 24;
      v17 = sub_BD2C40(72, unk_3F10A14);
      v3 = v17;
      if ( v17 )
        sub_B51BF0((__int64)v17, v12, *(_QWORD *)(a3 + 8), (__int64)v63, v16, 0);
      goto LABEL_10;
    }
    v40 = *(_QWORD *)(v61 + 8);
    v41 = *(unsigned __int8 *)(v40 + 8);
    if ( v41 == 17 )
    {
      v42 = 0;
    }
    else
    {
      if ( v41 != 18 )
        goto LABEL_5;
      v42 = 1;
    }
    v43 = *(_DWORD *)(v40 + 32);
    BYTE4(v62) = v42;
    v57 = v58;
    LODWORD(v62) = v43;
    v59 = v11;
    v44 = sub_BCE1B0((__int64 *)v13, (__int64)v62);
    v14 = v57;
    v11 = v59;
    v13 = v44;
    goto LABEL_5;
  }
  if ( v6 )
    v6 -= 24;
  v65 = 1;
  v63[0] = (__int64)"const_mat";
  v64 = 3;
  v3 = (_QWORD *)sub_B504D0(13, a2, v5, (__int64)v63, v6 + 24, 0);
LABEL_10:
  v18 = *(_QWORD *)(a3 + 32);
  v19 = v3 + 6;
  v20 = *(_QWORD *)(v18 + 48);
  v63[0] = v20;
  if ( !v20 )
  {
    if ( v19 == v63 )
      goto LABEL_15;
    v35 = v3[6];
    if ( !v35 )
      goto LABEL_15;
LABEL_48:
    sub_B91220((__int64)(v3 + 6), v35);
    goto LABEL_49;
  }
  sub_B96E90((__int64)v63, v20, 1);
  if ( v19 == v63 )
  {
    if ( v63[0] )
      sub_B91220((__int64)v63, v63[0]);
    goto LABEL_14;
  }
  v35 = v3[6];
  if ( v35 )
    goto LABEL_48;
LABEL_49:
  v36 = (unsigned __int8 *)v63[0];
  v3[6] = v63[0];
  if ( !v36 )
  {
LABEL_14:
    v18 = *(_QWORD *)(a3 + 32);
    goto LABEL_15;
  }
  sub_B976B0((__int64)v63, v36, (__int64)(v3 + 6));
  v18 = *(_QWORD *)(a3 + 32);
LABEL_15:
  v21 = *(_DWORD *)(a3 + 40);
  if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
    v22 = *(_QWORD *)(v18 - 8);
  else
    v22 = v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
  v23 = *(unsigned __int8 **)(v22 + 32LL * v21);
  result = *v23;
  if ( (_BYTE)result == 17 )
  {
    result = sub_272DBA0(v18, v21, (__int64)v3);
    if ( (_BYTE)result )
      return result;
    goto LABEL_43;
  }
  if ( (unsigned __int8)result > 0x1Cu )
  {
    v62 = *(_BYTE **)(v22 + 32LL * v21);
    v25 = (__int64 *)sub_C04EB0(a1 + 5576, (__int64 *)&v62);
    v26 = *v25;
    v27 = v25;
    if ( *v25 )
      return sub_272DBA0(*(_QWORD *)(a3 + 32), *(_DWORD *)(a3 + 40), v26);
    v47 = sub_B47F80(v62);
    *v27 = v47;
    if ( (*(_BYTE *)(v47 + 7) & 0x40) != 0 )
      v48 = *(_QWORD *)(v47 - 8);
    else
      v48 = v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v48 )
    {
      v49 = *(_QWORD *)(v48 + 8);
      **(_QWORD **)(v48 + 16) = v49;
      if ( v49 )
        *(_QWORD *)(v49 + 16) = *(_QWORD *)(v48 + 16);
    }
    *(_QWORD *)v48 = v3;
    if ( v3 )
    {
      v50 = v3[2];
      *(_QWORD *)(v48 + 8) = v50;
      if ( v50 )
        *(_QWORD *)(v50 + 16) = v48 + 8;
      *(_QWORD *)(v48 + 16) = v3 + 2;
      v3[2] = v48;
    }
    sub_B43E90(*v27, (__int64)(v62 + 24));
    v51 = *v27;
    v52 = *((_QWORD *)v62 + 6);
    v63[0] = v52;
    if ( v52 )
    {
      v53 = v51 + 48;
      sub_B96E90((__int64)v63, v52, 1);
      if ( (__int64 *)(v51 + 48) == v63 )
      {
        if ( v63[0] )
          sub_B91220(v51 + 48, v63[0]);
LABEL_84:
        v26 = *v27;
        return sub_272DBA0(*(_QWORD *)(a3 + 32), *(_DWORD *)(a3 + 40), v26);
      }
      v54 = *(_QWORD *)(v51 + 48);
      if ( !v54 )
      {
LABEL_91:
        v55 = (unsigned __int8 *)v63[0];
        *(_QWORD *)(v51 + 48) = v63[0];
        if ( v55 )
          sub_B976B0((__int64)v63, v55, v53);
        goto LABEL_84;
      }
    }
    else
    {
      v53 = v51 + 48;
      if ( (__int64 *)(v51 + 48) == v63 )
        goto LABEL_84;
      v54 = *(_QWORD *)(v51 + 48);
      if ( !v54 )
        goto LABEL_84;
    }
    sub_B91220(v53, v54);
    goto LABEL_91;
  }
  if ( *v23 == 5 )
  {
    if ( *((_WORD *)v23 + 1) == 34 )
      return sub_272DBA0(v18, v21, (__int64)v3);
    v28 = sub_AC5700(*(_QWORD *)(v22 + 32LL * v21));
    sub_B44220(v28, *(_QWORD *)(a3 + 16), *(_QWORD *)(a3 + 24));
    if ( (v28[7] & 0x40) != 0 )
      v29 = (unsigned __int8 *)*((_QWORD *)v28 - 1);
    else
      v29 = &v28[-32 * (*((_DWORD *)v28 + 1) & 0x7FFFFFF)];
    if ( *(_QWORD *)v29 )
    {
      v30 = *((_QWORD *)v29 + 1);
      **((_QWORD **)v29 + 2) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = *((_QWORD *)v29 + 2);
    }
    *(_QWORD *)v29 = v3;
    if ( v3 )
    {
      v31 = v3[2];
      *((_QWORD *)v29 + 1) = v31;
      if ( v31 )
        *(_QWORD *)(v31 + 16) = v29 + 8;
      *((_QWORD *)v29 + 2) = v3 + 2;
      v3[2] = v29;
    }
    v32 = *(_QWORD *)(a3 + 32);
    v33 = (__int64 *)(v28 + 48);
    v34 = *(_QWORD *)(v32 + 48);
    v63[0] = v34;
    if ( v34 )
    {
      sub_B96E90((__int64)v63, v34, 1);
      if ( v33 == v63 )
      {
        if ( v63[0] )
          sub_B91220((__int64)(v28 + 48), v63[0]);
        goto LABEL_40;
      }
      v45 = *((_QWORD *)v28 + 6);
      if ( !v45 )
      {
LABEL_69:
        v46 = (unsigned __int8 *)v63[0];
        *((_QWORD *)v28 + 6) = v63[0];
        if ( v46 )
        {
          sub_B976B0((__int64)v63, v46, (__int64)(v28 + 48));
          v32 = *(_QWORD *)(a3 + 32);
          goto LABEL_41;
        }
LABEL_40:
        v32 = *(_QWORD *)(a3 + 32);
LABEL_41:
        result = sub_272DBA0(v32, *(_DWORD *)(a3 + 40), (__int64)v28);
        if ( (_BYTE)result )
          return result;
        result = sub_B43D60(v28);
LABEL_43:
        if ( *(_QWORD *)a3 )
          return sub_B43D60(v3);
        return result;
      }
    }
    else
    {
      if ( v33 == v63 )
        goto LABEL_41;
      v45 = *((_QWORD *)v28 + 6);
      if ( !v45 )
        goto LABEL_41;
    }
    sub_B91220((__int64)(v28 + 48), v45);
    goto LABEL_69;
  }
  return result;
}
