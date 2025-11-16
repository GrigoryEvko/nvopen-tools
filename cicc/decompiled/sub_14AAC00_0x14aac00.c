// Function: sub_14AAC00
// Address: 0x14aac00
//
__int64 __fastcall sub_14AAC00(__int64 a1, unsigned __int64 a2, __int64 *a3, unsigned __int8 a4, unsigned int a5)
{
  __int64 *v5; // r14
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rdi
  unsigned int v11; // ebx
  __int64 v12; // rax
  unsigned __int8 v13; // cl
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rtt
  int v17; // eax
  unsigned int v18; // edx
  _QWORD *v19; // rax
  unsigned __int8 v20; // cl
  _QWORD *v21; // r15
  __int64 v22; // r9
  unsigned __int8 v23; // cl
  unsigned int v24; // r9d
  __int64 v25; // rax
  unsigned int v26; // ebx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // r9
  unsigned int v30; // r12d
  unsigned __int8 v31; // al
  __int64 v32; // r13
  unsigned int v33; // ebx
  __int64 v34; // r15
  unsigned int v35; // eax
  _QWORD *v36; // r10
  _QWORD *v37; // r9
  unsigned int v38; // eax
  __int64 v39; // r10
  _QWORD *v40; // r9
  __int64 *v42; // rax
  _QWORD *v43; // rbx
  unsigned __int8 v44; // al
  __int64 v45; // r13
  __int64 v46; // rsi
  unsigned int v47; // eax
  int v48; // eax
  unsigned int v49; // r9d
  __int64 v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rax
  unsigned int v53; // eax
  unsigned int v54; // eax
  _QWORD *v55; // rax
  unsigned __int64 v56; // [rsp+8h] [rbp-78h]
  unsigned __int8 v57; // [rsp+10h] [rbp-70h]
  unsigned __int8 v58; // [rsp+1Ch] [rbp-64h]
  unsigned __int8 v59; // [rsp+1Ch] [rbp-64h]
  unsigned int v60; // [rsp+1Ch] [rbp-64h]
  unsigned __int8 v61; // [rsp+20h] [rbp-60h]
  __int64 v62; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+20h] [rbp-60h]
  _QWORD *v64; // [rsp+20h] [rbp-60h]
  unsigned int v65; // [rsp+20h] [rbp-60h]
  unsigned int v66; // [rsp+20h] [rbp-60h]
  _QWORD *v67; // [rsp+20h] [rbp-60h]
  unsigned int v68; // [rsp+28h] [rbp-58h]
  unsigned int v69; // [rsp+28h] [rbp-58h]
  _QWORD *v70; // [rsp+28h] [rbp-58h]
  _QWORD *v71; // [rsp+28h] [rbp-58h]
  unsigned __int8 v72; // [rsp+28h] [rbp-58h]
  unsigned int v73; // [rsp+28h] [rbp-58h]
  _QWORD *v74; // [rsp+28h] [rbp-58h]
  unsigned __int64 v75; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v76; // [rsp+38h] [rbp-48h]
  __int64 v77; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v78; // [rsp+48h] [rbp-38h]

  while ( 1 )
  {
    if ( a5 > 5 )
      return 0;
    v5 = a3;
    v6 = *(_BYTE *)(a1 + 16);
    v7 = 0;
    v8 = a1;
    if ( v6 == 13 )
      v7 = a1;
    v9 = v7;
    if ( !(_DWORD)a2 )
      return 0;
    if ( (_DWORD)a2 == 1 )
    {
      *v5 = a1;
      return 1;
    }
    v10 = *(_QWORD *)a1;
    v58 = a4;
    v11 = a5;
    if ( v6 == 5 )
    {
      v12 = sub_15A0680(v10, (unsigned int)a2, 0);
      a2 = (unsigned int)a2;
      v13 = v58;
      if ( v8 == v12 )
      {
        v30 = 1;
        *v5 = sub_15A0680(v10, 1, 0);
        return v30;
      }
    }
    else
    {
      sub_15A0680(v10, (unsigned int)a2, 0);
      a2 = (unsigned int)a2;
      v13 = v58;
    }
    if ( v9 )
    {
      v14 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) > 0x40u )
        v14 = *(_QWORD *)v14;
      v16 = v14;
      v15 = v14 / a2;
      if ( !(v16 % a2) )
      {
        v30 = 1;
        *v5 = sub_15A0680(v10, v15, 0);
        return v30;
      }
    }
    v17 = *(unsigned __int8 *)(v8 + 16);
    if ( (unsigned __int8)v17 <= 0x17u )
    {
      if ( (_BYTE)v17 != 5 )
        return 0;
      v18 = *(unsigned __int16 *)(v8 + 18);
    }
    else
    {
      v18 = v17 - 24;
    }
    if ( v18 != 37 )
      break;
LABEL_38:
    v72 = v13;
    v42 = (__int64 *)sub_13CF970(v8);
    a5 = v11 + 1;
    a3 = v5;
    a1 = *v42;
    a4 = v72;
  }
  if ( v18 > 0x25 )
  {
    if ( v18 != 38 || !v13 )
      return 0;
    goto LABEL_38;
  }
  if ( ((v18 - 15) & 0xFFFFFFF7) != 0 )
    return 0;
  v61 = v13;
  v68 = v18;
  v19 = (_QWORD *)sub_13CF970(v8);
  v20 = v61;
  v21 = (_QWORD *)*v19;
  v22 = v19[3];
  if ( v68 != 23 )
    goto LABEL_24;
  if ( *(_BYTE *)(v22 + 16) != 13 )
    return 0;
  sub_13A38D0((__int64)&v75, v22 + 24);
  v23 = v61;
  v24 = v76 - 1;
  if ( v76 > 0x40 )
  {
    v57 = v61;
    v65 = v76;
    v60 = v76 - 1;
    v56 = v76 - 1;
    v48 = sub_16A57B0(&v75);
    v49 = v60;
    if ( v65 - v48 <= 0x40 && v56 >= *(_QWORD *)v75 )
      v49 = *(_QWORD *)v75;
    v78 = v65;
    v66 = v49;
    sub_16A4EF0(&v77, 0, 0);
    v23 = v57;
    v24 = v66;
  }
  else
  {
    v78 = v76;
    if ( v76 - 1 >= v75 )
      v24 = v75;
    v77 = 0;
  }
  v59 = v23;
  sub_14A9D60(&v77, v24);
  v25 = sub_16498A0(v8);
  v62 = sub_159C0E0(v25, &v77);
  sub_135E100(&v77);
  sub_135E100((__int64 *)&v75);
  v20 = v59;
  v22 = v62;
LABEL_24:
  v26 = v11 + 1;
  v63 = v22;
  v69 = v20;
  v75 = 0;
  v27 = sub_14AAC00(v21, (unsigned int)a2, &v75, v20, v26);
  v28 = v69;
  v29 = v63;
  v30 = v27;
  if ( (_BYTE)v27 )
  {
    v31 = *(_BYTE *)(v75 + 16);
    if ( *(_BYTE *)(v63 + 16) > 0x10u )
    {
      if ( v31 == 13 )
      {
        LOBYTE(v47) = sub_13A38F0(v75 + 24, (_QWORD *)1);
        v28 = v69;
        v29 = v63;
        v30 = v47;
        if ( (_BYTE)v47 )
        {
          *v5 = v63;
          return v30;
        }
      }
    }
    else
    {
      v70 = (_QWORD *)v75;
      if ( v31 <= 0x10u )
      {
        v32 = *(_QWORD *)v63;
        v33 = sub_1643030(*(_QWORD *)v63);
        v34 = *v70;
        v35 = sub_1643030(*v70);
        v36 = v70;
        v37 = (_QWORD *)v63;
        if ( v33 < v35 )
        {
          v51 = v63;
          v67 = v70;
          v52 = (_QWORD *)sub_15A3CB0(v51, v34, 0);
          v32 = *v52;
          v74 = v52;
          v53 = sub_1643030(*v52);
          v36 = v67;
          v37 = v74;
          v33 = v53;
        }
        v64 = v37;
        v71 = v36;
        v38 = sub_1643030(*v36);
        v39 = (__int64)v71;
        v40 = v64;
        if ( v38 < v33 )
        {
          v50 = sub_15A3CB0(v71, v32, 0);
          v40 = v64;
          v39 = v50;
        }
        *v5 = sub_15A2C20(v39, v40, 0, 0);
        return v30;
      }
    }
  }
  v77 = 0;
  v30 = sub_14AAC00(v29, (unsigned int)a2, &v77, v28, v26);
  if ( !(_BYTE)v30 )
    return 0;
  v43 = (_QWORD *)v77;
  v44 = *(_BYTE *)(v77 + 16);
  if ( *((_BYTE *)v21 + 16) > 0x10u )
  {
    if ( v44 == 13 )
    {
      LOBYTE(v54) = sub_13A38F0(v77 + 24, (_QWORD *)1);
      v30 = v54;
      if ( (_BYTE)v54 )
      {
        *v5 = (__int64)v21;
        return v30;
      }
    }
    return 0;
  }
  if ( v44 > 0x10u )
    return 0;
  v45 = *v21;
  v73 = sub_1643030(*v21);
  v46 = *v43;
  if ( v73 < (unsigned int)sub_1643030(*v43) )
  {
    v55 = (_QWORD *)sub_15A3CB0(v21, v46, 0);
    v45 = *v55;
    v21 = v55;
    v73 = sub_1643030(*v55);
  }
  if ( (unsigned int)sub_1643030(*v43) < v73 )
    v43 = (_QWORD *)sub_15A3CB0(v43, v45, 0);
  *v5 = sub_15A2C20(v43, v21, 0, 0);
  return v30;
}
