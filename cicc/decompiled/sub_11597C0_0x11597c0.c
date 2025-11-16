// Function: sub_11597C0
// Address: 0x11597c0
//
unsigned __int8 *__fastcall sub_11597C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // r13
  unsigned __int8 *v8; // rbx
  __int64 v9; // rdx
  unsigned __int8 *v10; // r13
  char v12; // cl
  unsigned __int8 v13; // si
  char v14; // al
  char v15; // si
  char v16; // al
  char v17; // cl
  unsigned int v18; // edx
  int v19; // eax
  char v20; // cl
  unsigned __int8 v21; // si
  bool v22; // al
  char v23; // si
  char v24; // al
  char v25; // r13
  bool v26; // al
  int v27; // eax
  char v28; // cl
  unsigned int v29; // edx
  __int64 v30; // r15
  __int64 v31; // r13
  unsigned int v32; // r14d
  unsigned int v33; // r14d
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // r15
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  int v43; // eax
  int v44; // eax
  __int64 v45; // rax
  unsigned __int8 *v46; // rax
  __int64 v47; // rax
  _BYTE *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  _BYTE *v53; // rax
  __int64 v54; // rdx
  _BYTE *v55; // rax
  const void *v56; // rdx
  unsigned __int64 v57; // rdx
  __int64 v58; // rdx
  _BYTE *v59; // rax
  __int64 v60; // rdx
  _BYTE *v61; // rax
  unsigned int v62; // [rsp+8h] [rbp-B8h]
  unsigned int v63; // [rsp+8h] [rbp-B8h]
  unsigned int v64; // [rsp+8h] [rbp-B8h]
  unsigned int v65; // [rsp+8h] [rbp-B8h]
  char v66; // [rsp+Ch] [rbp-B4h]
  char v67; // [rsp+Dh] [rbp-B3h]
  char v68; // [rsp+Eh] [rbp-B2h]
  char v69; // [rsp+Fh] [rbp-B1h]
  char v71; // [rsp+10h] [rbp-B0h]
  char v72; // [rsp+10h] [rbp-B0h]
  char v73; // [rsp+10h] [rbp-B0h]
  char v74; // [rsp+18h] [rbp-A8h]
  char v75; // [rsp+26h] [rbp-9Ah] BYREF
  char v76; // [rsp+27h] [rbp-99h] BYREF
  __int64 v77; // [rsp+28h] [rbp-98h] BYREF
  const void *v78; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v79; // [rsp+38h] [rbp-88h]
  unsigned __int64 v80; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v81; // [rsp+48h] [rbp-78h]
  const void *v82; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v83; // [rsp+58h] [rbp-68h]
  _BYTE v84[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v85; // [rsp+80h] [rbp-40h]

  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(unsigned __int8 **)(a2 - 64);
  if ( *(_BYTE *)v7 > 0x15u )
    goto LABEL_9;
  v9 = *v8;
  if ( (unsigned __int8)v9 <= 0x1Cu )
    goto LABEL_9;
  if ( (_BYTE)v9 == 86 )
  {
    v10 = sub_F26350(a1, (_BYTE *)a2, (__int64)v8, 0);
    if ( v10 )
      return v10;
    goto LABEL_7;
  }
  if ( (_BYTE)v9 == 84 )
  {
    if ( *(_BYTE *)v7 == 17 )
    {
      v31 = v7 + 24;
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
        goto LABEL_7;
      v48 = sub_AD7630(v7, 0, v9);
      if ( !v48 || *v48 != 17 )
        goto LABEL_7;
      v31 = (__int64)(v48 + 24);
    }
    v32 = *(_DWORD *)(v31 + 8);
    if ( v32 <= 0x40 )
    {
      if ( !*(_QWORD *)v31 || *(_BYTE *)a2 != 51 && 1LL << ((unsigned __int8)v32 - 1) == *(_QWORD *)v31 )
        goto LABEL_7;
    }
    else
    {
      if ( v32 == (unsigned int)sub_C444A0(v31) )
        goto LABEL_7;
      if ( *(_BYTE *)a2 != 51 )
      {
        v33 = v32 - 1;
        if ( (*(_QWORD *)(*(_QWORD *)v31 + 8LL * (v33 >> 6)) & (1LL << v33)) != 0
          && (unsigned int)sub_C44590(v31) == v33 )
        {
          goto LABEL_7;
        }
      }
    }
    v10 = sub_F27020(a1, a2, (__int64)v8, 0, a5, a6);
    if ( v10 )
      return v10;
  }
LABEL_7:
  v10 = (unsigned __int8 *)a2;
  if ( (unsigned __int8)sub_11AE870(a1, a2) )
    return v10;
  v8 = *(unsigned __int8 **)(a2 - 64);
  v7 = *(_QWORD *)(a2 - 32);
LABEL_9:
  v77 = 0;
  v79 = 1;
  v78 = 0;
  v81 = 1;
  v80 = 0;
  v75 = 1;
  v76 = 1;
  if ( (unsigned __int8)sub_1155C30(v8, &v77, (__int64)&v78, (bool *)&v75)
    && (unsigned __int8)sub_1155C30((_BYTE *)v7, &v77, (__int64)&v80, (bool *)&v76) )
  {
    v66 = 0;
    goto LABEL_12;
  }
  v30 = v77;
  if ( v77 )
    goto LABEL_35;
  if ( *v8 != 54 )
    goto LABEL_36;
  v34 = *((_QWORD *)v8 - 8);
  if ( *(_BYTE *)v34 == 17 )
  {
    v35 = *((_QWORD *)v8 - 4);
    v36 = v34 + 24;
    if ( !v35 )
      goto LABEL_36;
    goto LABEL_52;
  }
  v58 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v34 + 8) + 8LL) - 17;
  if ( (unsigned int)v58 > 1 || *(_BYTE *)v34 > 0x15u )
    goto LABEL_36;
  v59 = sub_AD7630(v34, 0, v58);
  if ( v59 )
  {
    if ( *v59 == 17 )
    {
      v36 = (__int64)(v59 + 24);
      v35 = *((_QWORD *)v8 - 4);
      if ( v35 )
      {
LABEL_52:
        v77 = v35;
        goto LABEL_53;
      }
    }
  }
  v30 = v77;
  if ( !v77 )
    goto LABEL_36;
LABEL_35:
  if ( *v8 != 54 )
    goto LABEL_36;
  v41 = *((_QWORD *)v8 - 8);
  v36 = v41 + 24;
  if ( *(_BYTE *)v41 != 17 )
  {
    v52 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v41 + 8) + 8LL) - 17;
    if ( (unsigned int)v52 > 1 )
      goto LABEL_36;
    if ( *(_BYTE *)v41 > 0x15u )
      goto LABEL_36;
    v53 = sub_AD7630(v41, 0, v52);
    if ( !v53 || *v53 != 17 )
      goto LABEL_36;
    v36 = (__int64)(v53 + 24);
  }
  v42 = *((_QWORD *)v8 - 4);
  if ( !v42 || v42 != v30 )
    goto LABEL_36;
LABEL_53:
  if ( v79 <= 0x40 && *(_DWORD *)(v36 + 8) <= 0x40u )
  {
    v56 = *(const void **)v36;
    v79 = *(_DWORD *)(v36 + 8);
    v78 = v56;
  }
  else
  {
    sub_C43990((__int64)&v78, v36);
  }
  v37 = v77;
  if ( v77 )
  {
LABEL_57:
    if ( *(_BYTE *)v7 == 54 )
    {
      v38 = *(_QWORD *)(v7 - 64);
      v39 = v38 + 24;
      if ( *(_BYTE *)v38 != 17 )
      {
        v54 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v38 + 8) + 8LL) - 17;
        if ( (unsigned int)v54 > 1 )
          goto LABEL_36;
        if ( *(_BYTE *)v38 > 0x15u )
          goto LABEL_36;
        v55 = sub_AD7630(v38, 0, v54);
        if ( !v55 || *v55 != 17 )
          goto LABEL_36;
        v39 = (__int64)(v55 + 24);
      }
      v40 = *(_QWORD *)(v7 - 32);
      if ( v40 && v40 == v37 )
        goto LABEL_61;
    }
LABEL_36:
    v77 = 0;
    v10 = 0;
    goto LABEL_37;
  }
  if ( *(_BYTE *)v7 != 54 )
    goto LABEL_36;
  v49 = *(_QWORD *)(v7 - 64);
  if ( *(_BYTE *)v49 == 17 )
  {
    v50 = *(_QWORD *)(v7 - 32);
    v39 = v49 + 24;
    if ( !v50 )
      goto LABEL_36;
  }
  else
  {
    v60 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v49 + 8) + 8LL) - 17;
    if ( (unsigned int)v60 > 1 || *(_BYTE *)v49 > 0x15u )
      goto LABEL_36;
    v61 = sub_AD7630(v49, 0, v60);
    if ( !v61 || *v61 != 17 || (v39 = (__int64)(v61 + 24), (v50 = *(_QWORD *)(v7 - 32)) == 0) )
    {
      v37 = v77;
      if ( !v77 )
        goto LABEL_36;
      goto LABEL_57;
    }
  }
  v77 = v50;
LABEL_61:
  if ( v81 <= 0x40 && *(_DWORD *)(v39 + 8) <= 0x40u )
  {
    v57 = *(_QWORD *)v39;
    v81 = *(_DWORD *)(v39 + 8);
    v80 = v57;
  }
  else
  {
    sub_C43990((__int64)&v80, v39);
  }
  v66 = 1;
LABEL_12:
  v12 = v75;
  v13 = v8[1];
  v14 = v13 >> 2;
  v15 = v13 >> 1;
  v16 = v14 & 1;
  v74 = *(_BYTE *)a2;
  if ( v75 )
    v12 = v16;
  v69 = v15 & 1;
  v68 = v12;
  if ( *(_BYTE *)a2 == 52 )
  {
    sub_C4B8A0((__int64)&v82, (__int64)&v78, (__int64)&v80);
    v17 = v68;
  }
  else
  {
    sub_C4B490((__int64)&v82, (__int64)&v78, (__int64)&v80);
    v17 = v15 & 1;
  }
  v18 = v83;
  if ( v83 <= 0x40 )
  {
    if ( v82 )
      goto LABEL_19;
  }
  else
  {
    v67 = v17;
    v62 = v83;
    v19 = sub_C444A0((__int64)&v82);
    v18 = v62;
    v17 = v67;
    if ( v62 != v19 )
      goto LABEL_19;
  }
  if ( v17 )
  {
    v45 = sub_AD6530(*(_QWORD *)(a2 + 8), (__int64)&v78);
    v46 = sub_F162A0(a1, a2, v45);
    v29 = v83;
    v10 = v46;
    goto LABEL_28;
  }
LABEL_19:
  v20 = v76;
  v21 = *(_BYTE *)(v7 + 1);
  v22 = (v21 & 4) != 0;
  v23 = v21 >> 1;
  if ( v76 )
    v20 = v22;
  v24 = v23 & 1;
  if ( v74 == 52 )
    v24 = v20;
  v25 = v24;
  if ( v18 <= 0x40 )
  {
    if ( v82 == v78 )
      goto LABEL_69;
    v65 = v18;
    v10 = 0;
    v73 = v20;
    v44 = sub_C49970((__int64)&v78, &v80);
    v28 = v73;
    v29 = v65;
    if ( v44 >= 0 )
    {
LABEL_26:
      if ( v74 != 52 )
      {
        v10 = 0;
        if ( !v69 )
          goto LABEL_28;
        goto LABEL_81;
      }
      if ( v68 && v28 )
      {
LABEL_81:
        v47 = sub_AD8D80(*(_QWORD *)(a2 + 8), (__int64)&v82);
        v85 = 257;
        if ( v66 )
          v10 = (unsigned __int8 *)sub_B504D0(25, v47, v77, (__int64)v84, 0, 0);
        else
          v10 = (unsigned __int8 *)sub_B504D0(17, v77, v47, (__int64)v84, 0, 0);
        sub_B44850(v10, 1);
        sub_B447F0(v10, v69);
        v29 = v83;
        goto LABEL_28;
      }
LABEL_71:
      v10 = 0;
      goto LABEL_28;
    }
LABEL_37:
    if ( v81 <= 0x40 )
      goto LABEL_31;
LABEL_38:
    if ( v80 )
      j_j___libc_free_0_0(v80);
    goto LABEL_31;
  }
  v63 = v18;
  v71 = v20;
  v26 = sub_C43C50((__int64)&v82, &v78);
  v20 = v71;
  v18 = v63;
  if ( !v26 )
  {
    v10 = 0;
    v27 = sub_C49970((__int64)&v78, &v80);
    v28 = v71;
    v29 = v63;
    if ( v27 < 0 )
      goto LABEL_29;
    goto LABEL_26;
  }
LABEL_69:
  v64 = v18;
  v72 = v20;
  if ( !v25 )
  {
    v43 = sub_C49970((__int64)&v78, &v80);
    v28 = v72;
    v29 = v64;
    if ( v43 < 0 )
      goto LABEL_71;
    goto LABEL_26;
  }
  v51 = sub_AD8D80(*(_QWORD *)(a2 + 8), (__int64)&v78);
  v85 = 257;
  if ( v66 )
    v10 = (unsigned __int8 *)sub_B504D0(25, v51, v77, (__int64)v84, 0, 0);
  else
    v10 = (unsigned __int8 *)sub_B504D0(17, v77, v51, (__int64)v84, 0, 0);
  sub_B44850(v10, v68 | (v74 == 52));
  sub_B447F0(v10, v69 | (v74 != 52));
  v29 = v83;
LABEL_28:
  if ( v29 <= 0x40 )
    goto LABEL_37;
LABEL_29:
  if ( !v82 )
    goto LABEL_37;
  j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 )
    goto LABEL_38;
LABEL_31:
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  return v10;
}
