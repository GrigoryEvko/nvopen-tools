// Function: sub_2D614B0
// Address: 0x2d614b0
//
__int64 __fastcall sub_2D614B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  char v6; // al
  __int64 v7; // r15
  __int64 v8; // rax
  bool v9; // zf
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v19; // r15
  __int64 *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  _BYTE *v23; // r10
  char **v24; // r14
  __int64 v25; // r8
  __int64 v26; // r9
  char **v27; // rax
  int v28; // r9d
  __int64 v29; // rax
  unsigned __int64 v30; // r11
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // r9d
  __int64 v34; // r15
  __int64 v35; // r14
  __int64 v36; // r8
  char v37; // al
  char *v38; // rax
  int v39; // eax
  __int64 v40; // rdi
  __int64 v41; // rax
  char *v42; // r14
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  char **v47; // r10
  __int64 v48; // rsi
  int v49; // edi
  __int64 v50; // r15
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // r15
  char *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rax
  char **v57; // r10
  char *v58; // r11
  __int64 v59; // rax
  __int64 v60; // rax
  _BYTE *v61; // [rsp+10h] [rbp-150h]
  int v62; // [rsp+1Ch] [rbp-144h]
  unsigned __int64 v63; // [rsp+20h] [rbp-140h]
  char v64; // [rsp+28h] [rbp-138h]
  __int64 v65; // [rsp+28h] [rbp-138h]
  char **v66; // [rsp+28h] [rbp-138h]
  _BYTE *v67; // [rsp+30h] [rbp-130h]
  __int64 v68; // [rsp+30h] [rbp-130h]
  _BYTE *v69; // [rsp+30h] [rbp-130h]
  char **v70; // [rsp+30h] [rbp-130h]
  __int64 *v71; // [rsp+30h] [rbp-130h]
  __int64 v72; // [rsp+30h] [rbp-130h]
  char **v73; // [rsp+30h] [rbp-130h]
  __int64 v74; // [rsp+30h] [rbp-130h]
  __int64 v75; // [rsp+30h] [rbp-130h]
  __int64 v76; // [rsp+38h] [rbp-128h]
  __int64 v77; // [rsp+38h] [rbp-128h]
  _BYTE *v78; // [rsp+38h] [rbp-128h]
  int v79; // [rsp+38h] [rbp-128h]
  __int64 v80; // [rsp+38h] [rbp-128h]
  __int64 v81; // [rsp+38h] [rbp-128h]
  char **v82; // [rsp+38h] [rbp-128h]
  char *v83; // [rsp+40h] [rbp-120h] BYREF
  _BYTE *v84; // [rsp+48h] [rbp-118h] BYREF
  char **v85; // [rsp+50h] [rbp-110h] BYREF
  __int64 v86; // [rsp+58h] [rbp-108h]
  _BYTE v87[16]; // [rsp+60h] [rbp-100h] BYREF
  _BYTE v88[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v89; // [rsp+90h] [rbp-D0h]
  unsigned int *v90[24]; // [rsp+A0h] [rbp-C0h] BYREF

  v6 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 > 0x1Cu )
  {
    if ( v6 != 63 )
      goto LABEL_3;
    v19 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
    if ( (unsigned int)v19 <= 1 || a2[5] != *(_QWORD *)(a3 + 40) )
    {
LABEL_20:
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    v21 = 32 * v19;
    v22 = sub_986520(a3);
    v23 = v87;
    v24 = (char **)v22;
    v25 = v22 + v21;
    v85 = (char **)v87;
    v26 = v21 >> 5;
    v86 = 0x200000000LL;
    v27 = (char **)v87;
    if ( v21 != 64 )
    {
      v65 = v25;
      sub_C8D5F0((__int64)&v85, v87, v21 >> 5, 8u, v25, v26);
      v25 = v65;
      LODWORD(v26) = v21 >> 5;
      v23 = v87;
      v27 = &v85[(unsigned int)v86];
    }
    do
    {
      if ( v27 )
        *v27 = *v24;
      v24 += 4;
      ++v27;
    }
    while ( (char **)v25 != v24 );
    LODWORD(v86) = v26 + v86;
    v28 = v86;
    v29 = (__int64)v85;
    v30 = (unsigned __int64)v85;
    v31 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)*v85 + 1) + 8LL) - 17;
    if ( (unsigned int)v31 > 1 )
    {
      v64 = 0;
    }
    else
    {
      v32 = sub_9B7920(*v85);
      v31 = (__int64)v85;
      v23 = v87;
      *v85 = (char *)v32;
      v29 = (__int64)v85;
      v30 = (unsigned __int64)v85;
      if ( !*v85 )
        goto LABEL_41;
      v64 = 1;
      v28 = v86;
    }
    v33 = v28 - 1;
    v34 = v33;
    if ( v33 > 1 )
    {
      v35 = 8;
      v36 = *(_QWORD *)(v30 + 8);
      v77 = 8LL * v33;
      v37 = *(_BYTE *)v36;
      if ( *(_BYTE *)v36 <= 0x15u )
      {
        while ( 1 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v36 + 8) + 8LL) - 17 <= 1 )
          {
            v67 = v23;
            v38 = sub_AD7630(v36, 0, v31);
            v30 = (unsigned __int64)v85;
            v23 = v67;
            v36 = (__int64)v38;
            if ( !v38 )
              goto LABEL_41;
            v37 = *v38;
          }
          if ( v37 != 17 )
            break;
          if ( *(_DWORD *)(v36 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v36 + 24) )
              goto LABEL_41;
          }
          else
          {
            v61 = v23;
            v62 = *(_DWORD *)(v36 + 32);
            v63 = v30;
            v68 = v36;
            v39 = sub_C444A0(v36 + 24);
            v36 = v68;
            v30 = v63;
            v23 = v61;
            if ( v62 != v39 )
              goto LABEL_41;
          }
          *(_QWORD *)(v30 + v35) = v36;
          v35 += 8;
          if ( v35 == v77 )
          {
            v29 = (__int64)v85;
            goto LABEL_47;
          }
          v30 = (unsigned __int64)v85;
          v36 = (__int64)v85[(unsigned __int64)v35 / 8];
          v37 = *(_BYTE *)v36;
          if ( *(_BYTE *)v36 > 0x15u )
            goto LABEL_41;
        }
      }
      goto LABEL_41;
    }
LABEL_47:
    v40 = *(_QWORD *)(v29 + 8 * v34);
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v40 + 8) + 8LL) - 17 <= 1 )
    {
      v78 = v23;
      v41 = sub_9B7920((char *)v40);
      v23 = v78;
      v42 = (char *)v41;
      if ( v41 )
      {
        if ( *(_BYTE *)v41 != 17 )
        {
LABEL_52:
          v85[v34] = v42;
          goto LABEL_53;
        }
        if ( *(_DWORD *)(v41 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v41 + 24) )
            goto LABEL_52;
        }
        else
        {
          v69 = v78;
          v79 = *(_DWORD *)(v41 + 32);
          v43 = sub_C444A0(v41 + 24);
          v23 = v69;
          if ( v79 != v43 )
            goto LABEL_52;
        }
      }
    }
    if ( v64 || (_DWORD)v86 != 2 )
    {
LABEL_53:
      v44 = *(_QWORD *)(a3 + 8);
      v70 = (char **)v23;
      v9 = *(_BYTE *)(v44 + 8) == 18;
      LODWORD(v44) = *(_DWORD *)(v44 + 32);
      BYTE4(v84) = v9;
      LODWORD(v84) = v44;
      sub_23D0AB0((__int64)v90, (__int64)a2, 0, 0, 0);
      v80 = *(_QWORD *)(a3 + 72);
      v45 = *((_QWORD *)*v85 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 <= 1 )
        v45 = **(_QWORD **)(v45 + 16);
      v66 = v70;
      v46 = sub_AE4570(*(_QWORD *)(a1 + 816), v45);
      v47 = v70;
      v71 = (__int64 *)v46;
      v48 = *((_QWORD *)v85[v34] + 1);
      v49 = *(unsigned __int8 *)(v48 + 8);
      if ( v49 == 18 )
      {
        v58 = *v85;
        v83 = v85[v34];
        if ( (_DWORD)v86 != 2 )
        {
LABEL_67:
          v74 = (__int64)v58;
          v60 = sub_AD6530(**(_QWORD **)(v48 + 16), v48);
          v85[v34] = (char *)v60;
          v89 = 257;
          v75 = sub_921130(v90, v80, v74, v85 + 1, (unsigned int)v86 - 1LL, (__int64)v88, 0);
          v80 = sub_B4DC50(v80, (__int64)(v85 + 1), (unsigned int)v86 - 1LL);
          v47 = v66;
          v58 = (char *)v75;
        }
      }
      else
      {
        if ( v49 != 17 )
        {
          v50 = v80;
          v89 = 257;
          v81 = sub_921130(v90, v80, (__int64)*v85, v85 + 1, (unsigned int)v86 - 1LL, (__int64)v88, 0);
          v72 = sub_BCE1B0(v71, (__int64)v84);
          v51 = (__int64)(v85 + 1);
          v52 = sub_B4DC50(v50, (__int64)(v85 + 1), (unsigned int)v86 - 1LL);
          v89 = 257;
          v53 = v52;
          v54 = (char *)sub_AD6530(v72, v51);
          v55 = v53;
          v83 = v54;
          v56 = sub_921130(v90, v53, v81, &v83, 1, (__int64)v88, 0);
          v57 = v66;
          v7 = v56;
          goto LABEL_58;
        }
        v58 = *v85;
        v83 = v85[v34];
        if ( (_DWORD)v86 != 2 )
          goto LABEL_67;
      }
      v55 = v80;
      v73 = v47;
      v89 = 257;
      v59 = sub_921130(v90, v80, (__int64)v58, &v83, 1, (__int64)v88, 0);
      v57 = v73;
      v7 = v59;
LABEL_58:
      v82 = v57;
      sub_F94A20(v90, v55);
      if ( v85 != v82 )
        _libc_free((unsigned __int64)v85);
      goto LABEL_13;
    }
    v30 = (unsigned __int64)v85;
LABEL_41:
    if ( (_BYTE *)v30 != v23 )
      _libc_free(v30);
    goto LABEL_20;
  }
  LODWORD(v7) = 0;
  if ( (unsigned __int8)v6 <= 0x15u )
    return (unsigned int)v7;
LABEL_3:
  v7 = sub_9B7920((char *)a3);
  if ( !v7 )
    return (unsigned int)v7;
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_BYTE *)(v8 + 8) == 18;
  LODWORD(v8) = *(_DWORD *)(v8 + 32);
  BYTE4(v85) = v9;
  LODWORD(v85) = v8;
  sub_23D0AB0((__int64)v90, (__int64)a2, 0, 0, 0);
  v10 = *(_QWORD *)(v7 + 8);
  v11 = *(_QWORD *)(a1 + 816);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v10 = **(_QWORD **)(v10 + 16);
  v12 = (__int64 *)sub_AE4570(v11, v10);
  v13 = sub_BCE1B0(v12, (__int64)v85);
  v14 = *(a2 - 4);
  if ( !v14 || *(_BYTE *)v14 || *(_QWORD *)(v14 + 24) != a2[10] )
    BUG();
  if ( *(_DWORD *)(v14 + 36) == 227 )
  {
    v16 = a2[1];
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v76 = v13;
  v15 = sub_986520((__int64)a2);
  v13 = v76;
  v16 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
LABEL_11:
    v16 = **(_QWORD **)(v16 + 16);
LABEL_12:
  v89 = 257;
  v84 = (_BYTE *)sub_AD6530(v13, v16);
  v7 = sub_921130(v90, v16, v7, &v84, 1, (__int64)v88, 0);
  sub_F94A20(v90, v16);
LABEL_13:
  v17 = v7;
  LODWORD(v7) = 1;
  sub_BD2ED0((__int64)a2, a3, v17);
  if ( !*(_QWORD *)(a3 + 16) )
  {
    v20 = *(__int64 **)(a1 + 48);
    v90[3] = (unsigned int *)sub_2D69720;
    v90[0] = (unsigned int *)a1;
    v90[2] = (unsigned int *)sub_2D56BA0;
    sub_F5CAB0((char *)a3, v20, 0, (__int64)v90);
    sub_A17130((__int64)v90);
  }
  return (unsigned int)v7;
}
