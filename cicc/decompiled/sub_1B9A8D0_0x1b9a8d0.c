// Function: sub_1B9A8D0
// Address: 0x1b9a8d0
//
__int64 __fastcall sub_1B9A8D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        double a5,
        double a6,
        double a7)
{
  _QWORD *v10; // r15
  unsigned __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // r13
  __int64 v28; // r15
  __int64 v29; // rax
  unsigned int *v30; // r8
  int v31; // r9d
  __int64 v32; // rsi
  __int64 v33; // r15
  __int64 v34; // rsi
  __int64 *v35; // r13
  _QWORD *v36; // r15
  unsigned int v37; // r13d
  unsigned int *v38; // r8
  int v39; // r9d
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rdi
  int v48; // eax
  __int64 v49; // rcx
  __int64 v50; // rsi
  int v51; // edi
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // r8
  __int64 v55; // r13
  unsigned __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  unsigned int v64; // esi
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  __int64 v67; // rcx
  int v68; // eax
  int v69; // r9d
  __int64 *v70; // [rsp+0h] [rbp-90h]
  __int64 v71; // [rsp+8h] [rbp-88h]
  __int64 v72; // [rsp+10h] [rbp-80h]
  __int64 v73; // [rsp+10h] [rbp-80h]
  __int64 v74; // [rsp+18h] [rbp-78h]
  unsigned int v75; // [rsp+24h] [rbp-6Ch]
  __int64 v76; // [rsp+28h] [rbp-68h]
  __int64 v77; // [rsp+28h] [rbp-68h]
  __int64 *v79; // [rsp+38h] [rbp-58h]
  __int64 v80[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v81; // [rsp+50h] [rbp-40h]

  v10 = *(_QWORD **)(a2 + 16);
  v79 = (__int64 *)(a1 + 96);
  v72 = *(_QWORD *)(a1 + 112);
  v74 = *(_QWORD *)(a1 + 104);
  v11 = sub_157EBA0(*(_QWORD *)(a1 + 168));
  sub_17050D0((__int64 *)(a1 + 96), v11);
  if ( *(_BYTE *)(a4 + 16) == 60 )
  {
    v67 = *(_QWORD *)a4;
    v81 = 257;
    v77 = v67;
    a3 = sub_12AA3B0(v79, 0x24u, a3, v67, (__int64)v80);
    v81 = 257;
    v10 = (_QWORD *)sub_12AA3B0(v79, 0x24u, (__int64)v10, v77, (__int64)v80);
  }
  v12 = *(_DWORD *)(a1 + 88);
  v81 = 257;
  v13 = sub_156DA60(v79, v12, v10, v80);
  v14 = *(_QWORD *)(a2 + 40);
  if ( v14 )
    v15 = (unsigned int)*(unsigned __int8 *)(v14 + 16) - 24;
  else
    v15 = 29;
  v16 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(*(_QWORD *)a1 + 24LL))(
          a1,
          v13,
          0,
          a3,
          v15);
  v17 = *(_QWORD *)a3;
  v70 = (__int64 *)v16;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 11 )
  {
    v75 = 11;
    v19 = 15;
    v20 = sub_15A0930(v17, *(unsigned int *)(a1 + 88));
  }
  else
  {
    v18 = *(_QWORD *)(a2 + 40);
    if ( v18 )
      v75 = *(unsigned __int8 *)(v18 + 16) - 24;
    else
      v75 = 29;
    a5 = (double)*(int *)(a1 + 88);
    v19 = 16;
    v20 = sub_15A10B0(v17, a5);
  }
  v81 = 257;
  v21 = sub_1904E90((__int64)v79, v19, a3, v20, v80, 0, a5, a6, a7);
  v22 = sub_1B8ED40(v21);
  if ( *(_BYTE *)(v22 + 16) > 0x10u )
  {
    v64 = *(_DWORD *)(a1 + 88);
    v81 = 257;
    v76 = sub_156DA60(v79, v64, (_QWORD *)v22, v80);
    v23 = v74;
    if ( v74 )
    {
LABEL_11:
      *(_QWORD *)(a1 + 104) = v23;
      *(_QWORD *)(a1 + 112) = v72;
      if ( v72 == v23 + 40 )
        goto LABEL_19;
      if ( !v72 )
        BUG();
      v24 = *(_QWORD *)(v72 + 24);
      v80[0] = v24;
      if ( v24 )
      {
        sub_1623A60((__int64)v80, v24, 2);
        v25 = *(_QWORD *)(a1 + 96);
        if ( !v25 )
          goto LABEL_16;
      }
      else
      {
        v25 = *(_QWORD *)(a1 + 96);
        if ( !v25 )
        {
LABEL_18:
          sub_17CD270(v80);
          goto LABEL_19;
        }
      }
      sub_161E7C0((__int64)v79, v25);
LABEL_16:
      v26 = (unsigned __int8 *)v80[0];
      *(_QWORD *)(a1 + 96) = v80[0];
      if ( v26 )
      {
        sub_1623210((__int64)v80, v26, (__int64)v79);
        v80[0] = 0;
      }
      goto LABEL_18;
    }
  }
  else
  {
    v76 = sub_15A0390(*(unsigned int *)(a1 + 88), v22);
    v23 = v74;
    if ( v74 )
      goto LABEL_11;
  }
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
LABEL_19:
  v27 = sub_157EE30(*(_QWORD *)(a1 + 200));
  v81 = 259;
  if ( v27 )
    v27 -= 24;
  v80[0] = (__int64)"vec.ind";
  v28 = *v70;
  v29 = sub_1648B60(64);
  v71 = v29;
  if ( v29 )
  {
    v32 = v28;
    v33 = v29;
    sub_15F1EA0(v29, v32, 53, 0, 0, v27);
    *(_DWORD *)(v33 + 56) = 2;
    sub_164B780(v33, v80);
    sub_1648880(v33, *(_DWORD *)(v33 + 56), 1);
  }
  v34 = *(_QWORD *)(a4 + 48);
  v80[0] = v34;
  v35 = (__int64 *)(v71 + 48);
  if ( !v34 )
  {
    if ( v35 == v80 )
      goto LABEL_27;
    v65 = *(_QWORD *)(v71 + 48);
    if ( !v65 )
      goto LABEL_27;
LABEL_51:
    sub_161E7C0((__int64)v35, v65);
    goto LABEL_52;
  }
  sub_1623A60((__int64)v80, v34, 2);
  if ( v35 == v80 )
  {
    if ( v80[0] )
      sub_161E7C0((__int64)v80, v80[0]);
    goto LABEL_27;
  }
  v65 = *(_QWORD *)(v71 + 48);
  if ( v65 )
    goto LABEL_51;
LABEL_52:
  v66 = (unsigned __int8 *)v80[0];
  *(_QWORD *)(v71 + 48) = v80[0];
  if ( v66 )
  {
    sub_1623210((__int64)v80, v66, (__int64)v35);
    if ( *(_DWORD *)(a1 + 92) )
      goto LABEL_28;
    goto LABEL_54;
  }
LABEL_27:
  if ( *(_DWORD *)(a1 + 92) )
  {
LABEL_28:
    v36 = (_QWORD *)v71;
    v37 = 0;
    sub_1B99BD0((unsigned int *)(a1 + 280), a4, 0, v71, v30, v31);
    while ( 1 )
    {
      if ( *(_BYTE *)(a4 + 16) == 60 )
        sub_1B91660(a1, (__int64)v36, a4);
      sub_1B9A880(a1, a2, a4, (__int64)v36, (unsigned int *)v37, (unsigned __int64 *)0xFFFFFFFFLL);
      v81 = 259;
      v80[0] = (__int64)"step.add";
      v41 = sub_1904E90((__int64)v79, v75, (__int64)v36, v76, v80, 0, a5, a6, a7);
      v42 = sub_1B8ED40(v41);
      v43 = *(_QWORD *)(a4 + 48);
      v36 = (_QWORD *)v42;
      v80[0] = v43;
      if ( !v43 )
        break;
      sub_1623A60((__int64)v80, v43, 2);
      v40 = (__int64)(v36 + 6);
      if ( v36 + 6 == v80 )
      {
        if ( v80[0] )
          sub_161E7C0((__int64)v80, v80[0]);
LABEL_32:
        if ( *(_DWORD *)(a1 + 92) <= ++v37 )
          goto LABEL_42;
        goto LABEL_33;
      }
      v44 = v36[6];
      if ( v44 )
        goto LABEL_39;
LABEL_40:
      v45 = (unsigned __int8 *)v80[0];
      v36[6] = v80[0];
      if ( !v45 )
        goto LABEL_32;
      ++v37;
      sub_1623210((__int64)v80, v45, v40);
      if ( *(_DWORD *)(a1 + 92) <= v37 )
        goto LABEL_42;
LABEL_33:
      sub_1B99BD0((unsigned int *)(a1 + 280), a4, v37, (__int64)v36, v38, v39);
    }
    v40 = v42 + 48;
    if ( (__int64 *)(v42 + 48) == v80 )
      goto LABEL_32;
    v44 = *(_QWORD *)(v42 + 48);
    if ( !v44 )
      goto LABEL_32;
LABEL_39:
    v73 = v40;
    sub_161E7C0(v40, v44);
    v40 = v73;
    goto LABEL_40;
  }
LABEL_54:
  v36 = (_QWORD *)v71;
LABEL_42:
  v46 = *(_QWORD *)(a1 + 24);
  v47 = 0;
  v48 = *(_DWORD *)(v46 + 24);
  if ( v48 )
  {
    v49 = *(_QWORD *)(a1 + 200);
    v50 = *(_QWORD *)(v46 + 8);
    v51 = v48 - 1;
    v52 = (v48 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
    v53 = (__int64 *)(v50 + 16LL * v52);
    v54 = *v53;
    if ( v49 == *v53 )
    {
LABEL_44:
      v47 = v53[1];
    }
    else
    {
      v68 = 1;
      while ( v54 != -8 )
      {
        v69 = v68 + 1;
        v52 = v51 & (v68 + v52);
        v53 = (__int64 *)(v50 + 16LL * v52);
        v54 = *v53;
        if ( v49 == *v53 )
          goto LABEL_44;
        v68 = v69;
      }
      v47 = 0;
    }
  }
  v55 = sub_13FCB50(v47);
  v56 = sub_157EBA0(v55);
  sub_15F22F0(v36, *(_QWORD *)(v56 - 72));
  v81 = 259;
  v80[0] = (__int64)"vec.ind.next";
  sub_164B780((__int64)v36, v80);
  sub_1704F80(v71, (__int64)v70, *(_QWORD *)(a1 + 168), v57, v58, v59);
  return sub_1704F80(v71, (__int64)v36, v55, v60, v61, v62);
}
