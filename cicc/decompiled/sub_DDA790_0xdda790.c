// Function: sub_DDA790
// Address: 0xdda790
//
__int64 __fastcall sub_DDA790(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7)
{
  unsigned __int64 v8; // r14
  __int64 v10; // r10
  int v12; // eax
  bool v13; // zf
  __int64 v14; // rdx
  __int64 v15; // r8
  __int16 v16; // ax
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // r9d
  __int64 *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // r15
  char v25; // al
  __int64 v26; // r10
  unsigned int v27; // eax
  char v28; // al
  unsigned int v29; // eax
  __int64 v31; // rdx
  __int64 v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // rax
  unsigned int v35; // r14d
  _BYTE *v36; // rsi
  __int64 *v37; // r15
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  _QWORD *v41; // r15
  _QWORD *v42; // rax
  char v43; // al
  __int64 v44; // r10
  char v45; // al
  _QWORD *v46; // rax
  _QWORD *v47; // r15
  char v48; // al
  char v49; // al
  char v50; // al
  char v51; // al
  unsigned __int8 v52; // [rsp+7h] [rbp-89h]
  __int64 v53; // [rsp+8h] [rbp-88h]
  _QWORD *v54; // [rsp+8h] [rbp-88h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  unsigned __int8 v58; // [rsp+10h] [rbp-80h]
  __int64 v59; // [rsp+18h] [rbp-78h]
  __int64 v60; // [rsp+18h] [rbp-78h]
  __int64 v61; // [rsp+18h] [rbp-78h]
  unsigned __int8 v62; // [rsp+18h] [rbp-78h]
  _QWORD *v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+20h] [rbp-70h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  __int64 v66; // [rsp+20h] [rbp-70h]
  __int64 v67; // [rsp+20h] [rbp-70h]
  __int64 v68; // [rsp+20h] [rbp-70h]
  __int64 v70; // [rsp+20h] [rbp-70h]
  __int64 v71; // [rsp+20h] [rbp-70h]
  __int64 v72; // [rsp+20h] [rbp-70h]
  __int64 v73; // [rsp+28h] [rbp-68h] BYREF
  __int64 v74; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v75; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v76; // [rsp+48h] [rbp-48h]
  __int64 *v77; // [rsp+50h] [rbp-40h]
  unsigned int *v78; // [rsp+58h] [rbp-38h]

  v73 = a6;
  if ( a7 > (unsigned int)qword_4F894C8 )
    return 0;
  v8 = HIDWORD(a2);
  v10 = a4;
  v12 = a2;
  if ( (((_DWORD)a2 - 36) & 0xFFFFFFFB) == 0 )
  {
    v12 = sub_B52F50(a2);
    v31 = v73;
    v73 = a5;
    a5 = v31;
    v32 = a3;
    a3 = a4;
    v10 = v32;
  }
  if ( v12 == 34 )
  {
    v68 = v10;
    if ( !(unsigned __int8)sub_DBED40((__int64)a1, a5) )
      return 0;
    if ( !(unsigned __int8)sub_DBED40((__int64)a1, v73) )
      return 0;
    v34 = sub_D95540(a3);
    v35 = (unsigned int)sub_DA2C50((__int64)a1, v34, -1, 1u);
    if ( !(unsigned __int8)sub_DDB0E0((_DWORD)a1, 38, a3, v35, a5, v73, 0)
      || !(unsigned __int8)sub_DDB0E0((_DWORD)a1, 38, v68, v35, a5, v73, 0) )
    {
      return 0;
    }
    v10 = v68;
    LOBYTE(v8) = 0;
  }
  else if ( v12 != 38 )
  {
    return 0;
  }
  v13 = *(_WORD *)(a3 + 24) == 4;
  v74 = a5;
  v14 = a3;
  if ( v13 )
    v14 = *(_QWORD *)(a3 + 32);
  v15 = a5;
  if ( *(_WORD *)(a5 + 24) == 4 )
    v15 = *(_QWORD *)(a5 + 32);
  v75 = a1;
  v76 = &v74;
  v77 = &v73;
  v78 = &a7;
  v16 = *(_WORD *)(v14 + 24);
  if ( v16 == 5 )
  {
    v59 = v14;
    v64 = v10;
    v17 = sub_D95540(v14);
    v18 = sub_D97050((__int64)a1, v17);
    v19 = sub_D95540(v64);
    v20 = sub_D97050((__int64)a1, v19);
    v21 = 0;
    if ( v18 == v20 && (*(_BYTE *)(v59 + 28) & 4) != 0 )
    {
      v22 = *(__int64 **)(v59 + 32);
      v60 = *v22;
      v56 = v22[1];
      v23 = sub_D95540(v64);
      v24 = sub_DA2C50((__int64)a1, v23, -1, 1u);
      v25 = sub_DCD020(v75, 38, v60, (__int64)v24);
      v26 = v64;
      if ( !v25 )
      {
        v50 = sub_DDA790((_DWORD)v75, 38, v60, (_DWORD)v24, *v76, *v77, *v78 + 1);
        v26 = v64;
        if ( !v50 )
          goto LABEL_49;
      }
      v65 = v26;
      v21 = sub_DCD020(v75, 38, v56, v26);
      if ( !(_BYTE)v21 )
      {
        v27 = sub_DDA790((_DWORD)v75, 38, v56, v65, *v76, *v77, *v78 + 1);
        v26 = v65;
        v21 = v27;
        if ( !(_BYTE)v27 )
        {
LABEL_49:
          v66 = v26;
          v28 = sub_DCD020(v75, 38, v56, (__int64)v24);
          v10 = v66;
          if ( !v28 )
          {
            v51 = sub_DDA790((_DWORD)v75, 38, v56, (_DWORD)v24, *v76, *v77, *v78 + 1);
            v10 = v66;
            if ( !v51 )
              goto LABEL_19;
          }
          v67 = v10;
          v21 = sub_DCD020(v75, 38, v60, v10);
          if ( !(_BYTE)v21 )
          {
            v29 = sub_DDA790((_DWORD)v75, 38, v60, v67, *v76, *v77, *v78 + 1);
            v10 = v67;
            v21 = v29;
            if ( !(_BYTE)v29 )
            {
LABEL_19:
              a5 = v74;
              return (unsigned int)sub_DD9D50(
                                     (__int64)a1,
                                     ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                                     a3,
                                     v10,
                                     a5,
                                     v73,
                                     a7 + 1);
            }
          }
        }
      }
    }
    return v21;
  }
  if ( v16 != 15 )
    return (unsigned int)sub_DD9D50(
                           (__int64)a1,
                           ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                           a3,
                           v10,
                           a5,
                           v73,
                           a7 + 1);
  v33 = *(_BYTE **)(v14 - 8);
  if ( *v33 != 49 )
    return (unsigned int)sub_DD9D50(
                           (__int64)a1,
                           ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                           a3,
                           v10,
                           a5,
                           v73,
                           a7 + 1);
  if ( !*((_QWORD *)v33 - 8) )
    return (unsigned int)sub_DD9D50(
                           (__int64)a1,
                           ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                           a3,
                           v10,
                           a5,
                           v73,
                           a7 + 1);
  v36 = (_BYTE *)*((_QWORD *)v33 - 4);
  if ( !v36 )
    return (unsigned int)sub_DD9D50(
                           (__int64)a1,
                           ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                           a3,
                           v10,
                           a5,
                           v73,
                           a7 + 1);
  if ( *v36 != 17 )
    return 0;
  v70 = *((_QWORD *)v33 - 8);
  v57 = v10;
  v61 = v15;
  v37 = sub_DD8400((__int64)a1, (__int64)v36);
  v38 = sub_D98300((__int64)a1, v70);
  if ( !v38 )
    return 0;
  v53 = v38;
  v71 = sub_D95540(v38);
  if ( v71 != sub_D95540(v61) )
    return 0;
  if ( !sub_D90F00(v53, v61) )
    return 0;
  v62 = sub_DBEDC0((__int64)a1, (__int64)v37);
  if ( !v62 )
    return 0;
  v39 = *(_QWORD *)(v37[4] + 8);
  v40 = sub_D95540(v73);
  if ( (*(_BYTE *)(v40 + 8) == 14) != (*(_BYTE *)(v39 + 8) == 14) )
    return 0;
  v52 = v62;
  v72 = sub_D970B0((__int64)a1, v39, v40);
  v41 = sub_DD2D10((__int64)a1, (__int64)v37, v72);
  v63 = sub_DD2D10((__int64)a1, v73, v72);
  v42 = sub_DA2C50((__int64)a1, v72, 2, 0);
  v54 = sub_DCC810(a1, (__int64)v41, (__int64)v42, 0, 0);
  v43 = sub_DBEC80((__int64)a1, v57);
  v44 = v57;
  LOBYTE(v21) = v52;
  if ( !v43 || (v45 = sub_DDAE80(&v75, v63, v54), v21 = v52, v44 = v57, !v45) )
  {
    v58 = v21;
    v55 = v44;
    v46 = sub_DA2C50((__int64)a1, v72, -1, 1u);
    v47 = sub_DCC810(a1, (__int64)v46, (__int64)v41, 0, 0);
    v48 = sub_DBEC00((__int64)a1, v55);
    v10 = v55;
    if ( v48 )
    {
      v49 = sub_DDAE80(&v75, v63, v47);
      v21 = v58;
      if ( v49 )
        return v21;
      a5 = v74;
      v10 = v55;
      return (unsigned int)sub_DD9D50(
                             (__int64)a1,
                             ((unsigned __int64)(unsigned __int8)v8 << 32) | a2 & 0xFFFFFF0000000000LL | 0x26,
                             a3,
                             v10,
                             a5,
                             v73,
                             a7 + 1);
    }
    goto LABEL_19;
  }
  return v21;
}
