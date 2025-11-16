// Function: sub_112A9F0
// Address: 0x112a9f0
//
unsigned __int8 *__fastcall sub_112A9F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // r15
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rax
  unsigned __int8 *v11; // r14
  unsigned __int8 v12; // al
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  __int64 v16; // r14
  int v17; // eax
  __int64 v18; // rdi
  _BYTE *v19; // r14
  __int64 v20; // rdx
  _BYTE *v21; // r14
  __int64 v22; // rdi
  int v23; // ebx
  _BYTE *v24; // r14
  unsigned __int64 v25; // rax
  int *v26; // rdx
  __int64 v27; // rax
  int *v28; // rdx
  int v29; // r15d
  int v30; // r14d
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // ebx
  __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned int **v38; // r13
  _BYTE *v39; // rax
  __int64 v40; // rax
  unsigned int **v41; // r13
  _BYTE *v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // r15
  __int64 v45; // rax
  unsigned __int64 v46; // rax
  int *v47; // rdx
  unsigned __int64 v48; // rax
  int *v49; // rdx
  unsigned __int64 v50; // rax
  int *v51; // rdx
  unsigned __int64 v52; // rax
  int *v53; // rdx
  __int64 v54; // [rsp+10h] [rbp-150h]
  bool v55; // [rsp+18h] [rbp-148h]
  bool v56; // [rsp+77h] [rbp-E9h] BYREF
  unsigned __int8 *v57; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v58; // [rsp+80h] [rbp-E0h] BYREF
  unsigned __int8 *v59; // [rsp+88h] [rbp-D8h] BYREF
  int v60; // [rsp+90h] [rbp-D0h] BYREF
  char v61; // [rsp+94h] [rbp-CCh]
  int v62; // [rsp+98h] [rbp-C8h] BYREF
  char v63; // [rsp+9Ch] [rbp-C4h]
  _QWORD v64[4]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v65; // [rsp+C0h] [rbp-A0h]
  int *v66; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v67; // [rsp+D8h] [rbp-88h]
  unsigned __int8 **v68; // [rsp+E0h] [rbp-80h]
  int *v69; // [rsp+E8h] [rbp-78h]
  __int64 *v70; // [rsp+F0h] [rbp-70h]
  __int64 *v71; // [rsp+F8h] [rbp-68h] BYREF
  char v72; // [rsp+100h] [rbp-60h]
  int *v73; // [rsp+108h] [rbp-58h]
  __int64 *v74; // [rsp+110h] [rbp-50h]
  __int64 *v75; // [rsp+118h] [rbp-48h] BYREF
  char v76; // [rsp+120h] [rbp-40h]

  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(unsigned __int8 **)(a2 - 32);
  if ( !v4 )
  {
    if ( v5 )
    {
      v57 = *(unsigned __int8 **)(a2 - 32);
      BUG();
    }
    goto LABEL_6;
  }
  v6 = *((_QWORD *)v5 + 2);
  v57 = *(unsigned __int8 **)(a2 - 64);
  if ( v6 )
  {
    if ( !*(_QWORD *)(v6 + 8) && *v5 == 68 )
    {
      v14 = *((_QWORD *)v5 - 4);
      if ( v14 )
      {
        v15 = sub_B53900(a2) & 0xFFFFFFFFFFLL;
        goto LABEL_27;
      }
    }
  }
  v7 = *(_QWORD *)(v4 + 16);
  v57 = v5;
  if ( v7 )
  {
    if ( !*(_QWORD *)(v7 + 8) && *(_BYTE *)v4 == 68 )
    {
      v14 = *(_QWORD *)(v4 - 32);
      if ( v14 )
      {
        v15 = sub_B53960(a2) & 0xFFFFFFFFFFLL;
LABEL_27:
        v18 = *(_QWORD *)(v14 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
          v18 = **(_QWORD **)(v18 + 16);
        if ( sub_BCAC40(v18, 1) && (_DWORD)v15 == 36 )
        {
          v38 = *(unsigned int ***)(a1 + 32);
          LOWORD(v70) = 257;
          v65 = 257;
          v39 = (_BYTE *)sub_AD6530(*((_QWORD *)v57 + 1), 1);
          v40 = sub_92B530(v38, 0x20u, (__int64)v57, v39, (__int64)v64);
          return (unsigned __int8 *)sub_B504D0(28, v40, v14, (__int64)&v66, 0, 0);
        }
        v4 = *(_QWORD *)(a2 - 64);
        v5 = *(unsigned __int8 **)(a2 - 32);
        if ( !v4 )
        {
          if ( v5 )
          {
            v57 = *(unsigned __int8 **)(a2 - 32);
            BUG();
          }
          goto LABEL_6;
        }
      }
    }
  }
  v8 = *((_QWORD *)v5 + 2);
  v57 = (unsigned __int8 *)v4;
  if ( v8 && !*(_QWORD *)(v8 + 8) && *v5 == 69 && (v16 = *((_QWORD *)v5 - 4)) != 0 )
  {
    v17 = sub_B53900(a2);
  }
  else
  {
    v9 = *(_QWORD *)(v4 + 16);
    v57 = v5;
    if ( !v9 )
      goto LABEL_6;
    if ( *(_QWORD *)(v9 + 8) )
      goto LABEL_6;
    if ( *(_BYTE *)v4 != 69 )
      goto LABEL_6;
    v16 = *(_QWORD *)(v4 - 32);
    if ( !v16 )
      goto LABEL_6;
    v17 = sub_B53960(a2);
  }
  v22 = *(_QWORD *)(v16 + 8);
  v23 = v17;
  if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
    v22 = **(_QWORD **)(v22 + 16);
  if ( sub_BCAC40(v22, 1) && v23 == 37 )
  {
    v41 = *(unsigned int ***)(a1 + 32);
    v65 = 257;
    LOWORD(v70) = 257;
    v42 = (_BYTE *)sub_AD6530(*((_QWORD *)v57 + 1), 1);
    v43 = sub_92B530(v41, 0x20u, (__int64)v57, v42, (__int64)v64);
    return (unsigned __int8 *)sub_B504D0(29, v43, v16, (__int64)&v66, 0, 0);
  }
  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(unsigned __int8 **)(a2 - 32);
LABEL_6:
  v60 = 42;
  v68 = &v59;
  v61 = 0;
  v62 = 42;
  v63 = 0;
  v66 = &v60;
  v67 = (__int64 *)&v57;
  v69 = &v62;
  v70 = (__int64 *)&v57;
  v71 = &v58;
  v72 = 0;
  v73 = &v62;
  v74 = (__int64 *)&v57;
  v75 = &v58;
  v76 = 0;
  if ( !v4 )
    goto LABEL_41;
  v57 = (unsigned __int8 *)v4;
  if ( *v5 <= 0x1Cu )
  {
LABEL_10:
    v10 = (__int64 *)&v57;
    goto LABEL_11;
  }
  v59 = v5;
  if ( *v5 == 68 )
  {
    v19 = (_BYTE *)*((_QWORD *)v5 - 4);
    if ( *v19 != 82 || (v20 = *((_QWORD *)v19 - 8)) == 0 || v20 != v4 )
    {
LABEL_42:
      v10 = v67;
LABEL_11:
      *v10 = (__int64)v5;
      v11 = *(unsigned __int8 **)(a2 - 64);
      if ( *v11 <= 0x1Cu )
        return 0;
      *v68 = v11;
      v12 = *v11;
      if ( *v11 == 68 )
      {
        v44 = (_BYTE *)*((_QWORD *)v11 - 4);
        if ( *v44 != 82 || *((_QWORD *)v44 - 8) != *v70 )
          return 0;
        if ( (unsigned __int8)sub_991580((__int64)&v71, *((_QWORD *)v44 - 4)) )
        {
          if ( v69 )
          {
            v52 = sub_B53900((__int64)v44);
            v53 = v69;
            *v69 = v52;
            *((_BYTE *)v53 + 4) = BYTE4(v52);
          }
LABEL_57:
          if ( v66 )
          {
            v27 = sub_B53960(a2);
            v28 = v66;
            *v66 = v27;
            *((_BYTE *)v28 + 4) = BYTE4(v27);
          }
          goto LABEL_59;
        }
        v12 = *v11;
      }
      if ( v12 != 69 )
        return 0;
      v24 = (_BYTE *)*((_QWORD *)v11 - 4);
      if ( *v24 != 82
        || *((_QWORD *)v24 - 8) != *v74
        || !(unsigned __int8)sub_991580((__int64)&v75, *((_QWORD *)v24 - 4)) )
      {
        return 0;
      }
      if ( v73 )
      {
        v25 = sub_B53900((__int64)v24);
        v26 = v73;
        *v73 = v25;
        *((_BYTE *)v26 + 4) = BYTE4(v25);
      }
      goto LABEL_57;
    }
    if ( (unsigned __int8)sub_991580((__int64)&v71, *((_QWORD *)v19 - 4)) )
    {
      if ( v69 )
      {
        v50 = sub_B53900((__int64)v19);
        v51 = v69;
        *v69 = v50;
        *((_BYTE *)v51 + 4) = BYTE4(v50);
      }
      goto LABEL_98;
    }
    if ( *v5 != 69 )
      goto LABEL_40;
  }
  else if ( *v5 != 69 )
  {
    goto LABEL_10;
  }
  v21 = (_BYTE *)*((_QWORD *)v5 - 4);
  if ( *v21 != 82 || *((_QWORD *)v21 - 8) != *v74 || !(unsigned __int8)sub_991580((__int64)&v75, *((_QWORD *)v21 - 4)) )
  {
LABEL_40:
    v5 = *(unsigned __int8 **)(a2 - 32);
LABEL_41:
    if ( !v5 )
      return 0;
    goto LABEL_42;
  }
  if ( v73 )
  {
    v46 = sub_B53900((__int64)v21);
    v47 = v73;
    *v73 = v46;
    *((_BYTE *)v47 + 4) = BYTE4(v46);
  }
LABEL_98:
  if ( v66 )
  {
    v48 = sub_B53900(a2);
    v49 = v66;
    *v66 = v48;
    *((_BYTE *)v49 + 4) = BYTE4(v48);
  }
LABEL_59:
  v29 = v60;
  if ( (unsigned int)(v60 - 32) > 1 )
    return 0;
  v30 = v62;
  if ( (unsigned int)(v62 - 32) > 1 )
    return 0;
  v55 = 0;
  v31 = *v59;
  v32 = *((_QWORD *)v59 + 2);
  v56 = (_BYTE)v31 == 69;
  if ( v32 )
  {
    if ( !*(_QWORD *)(v32 + 8) )
    {
      v31 = (unsigned __int8)v31;
      v37 = *(_QWORD *)(*(_QWORD *)sub_986520((__int64)v59) + 16LL);
      if ( v37 )
        v55 = *(_QWORD *)(v37 + 8) == 0;
    }
  }
  v64[0] = a1;
  v64[1] = &v60;
  v64[2] = &v57;
  v64[3] = &v56;
  v33 = *(_DWORD *)(v58 + 8);
  if ( v33 > 0x40 )
  {
    v54 = v58;
    v31 = (unsigned __int8)v31;
    if ( v33 != (unsigned int)sub_C444A0(v58) )
    {
      if ( (_BYTE)v31 != 69 )
      {
        v31 = (unsigned __int8)v31;
        if ( (unsigned int)sub_C444A0(v54) != v33 - 1 )
          goto LABEL_66;
LABEL_72:
        if ( v30 != 33 )
          return (unsigned __int8 *)sub_1111ED0((__int64)v64, v31);
LABEL_89:
        v45 = sub_AD64A0(*(_QWORD *)(a2 + 8), v29 == 33);
        return sub_F162A0(a1, a2, v45);
      }
      v31 = 69;
      if ( v33 != (unsigned int)sub_C445E0(v54) )
        goto LABEL_66;
LABEL_88:
      if ( v30 == 33 )
        goto LABEL_89;
LABEL_76:
      if ( v55 )
        return (unsigned __int8 *)sub_1111ED0((__int64)v64, v31);
      return 0;
    }
LABEL_74:
    if ( v30 == 32 )
      goto LABEL_89;
    if ( (_BYTE)v31 != 69 )
      return (unsigned __int8 *)sub_1111ED0((__int64)v64, v31);
    goto LABEL_76;
  }
  if ( !*(_QWORD *)v58 )
    goto LABEL_74;
  if ( (_BYTE)v31 != 69 )
  {
    if ( *(_QWORD *)v58 != 1 )
      goto LABEL_66;
    goto LABEL_72;
  }
  if ( !v33 || *(_QWORD *)v58 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) )
    goto LABEL_88;
LABEL_66:
  v34 = 0;
  LOWORD(v70) = 257;
  if ( v30 == 33 )
    v34 = 2LL * ((_BYTE)v31 != 69) - 1;
  v35 = sub_AD64C0(*((_QWORD *)v57 + 1), v34, 1u);
  return (unsigned __int8 *)sub_B52500(53, v60, (__int64)v57, v35, (__int64)&v66, v36, 0, 0);
}
