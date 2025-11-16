// Function: sub_247AF50
// Address: 0x247af50
//
void __fastcall sub_247AF50(__int64 *a1, unsigned __int8 *a2, int a3)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // r14
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  int v11; // edx
  int v12; // r14d
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // eax
  _BYTE *v20; // rcx
  int v21; // eax
  unsigned int v22; // r14d
  __int64 v23; // rdx
  unsigned int v24; // ebx
  unsigned __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r9
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // edx
  unsigned __int64 v32; // r14
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rbx
  _BYTE *v40; // rax
  __int64 v41; // rax
  _BYTE *v42; // rbx
  _BYTE *v43; // rax
  _BYTE *v44; // rdx
  unsigned __int64 v45; // r14
  _QWORD *v46; // rax
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rax
  __int64 v49; // [rsp+8h] [rbp-188h]
  int v50; // [rsp+8h] [rbp-188h]
  __int64 v51; // [rsp+8h] [rbp-188h]
  __int64 **v52; // [rsp+10h] [rbp-180h]
  __int64 v53; // [rsp+10h] [rbp-180h]
  __int64 v54; // [rsp+10h] [rbp-180h]
  _BYTE *v55; // [rsp+10h] [rbp-180h]
  __int64 v56; // [rsp+20h] [rbp-170h]
  int v57; // [rsp+20h] [rbp-170h]
  int v58; // [rsp+38h] [rbp-158h]
  _BYTE v59[32]; // [rsp+40h] [rbp-150h] BYREF
  __int16 v60; // [rsp+60h] [rbp-130h]
  _BYTE *v61; // [rsp+70h] [rbp-120h] BYREF
  __int64 v62; // [rsp+78h] [rbp-118h]
  _BYTE v63[32]; // [rsp+80h] [rbp-110h] BYREF
  _BYTE *v64; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v65; // [rsp+A8h] [rbp-E8h]
  _BYTE v66[32]; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int *v67[24]; // [rsp+D0h] [rbp-C0h] BYREF

  v3 = a3;
  v4 = *(_QWORD *)(*(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] + 8LL);
  sub_23D0AB0((__int64)v67, (__int64)a2, 0, 0, 0);
  sub_A17190(a2);
  v5 = sub_BCAE30(v4);
  v65 = v6;
  v64 = (_BYTE *)v5;
  v7 = sub_CA1930(&v64);
  v8 = (__int64 *)sub_BCD140((_QWORD *)v67[9], v3);
  v9 = sub_BCDA70(v8, v7 / v3);
  v11 = *a2;
  v52 = (__int64 **)v9;
  v12 = *(_DWORD *)(v9 + 32);
  if ( v11 == 40 )
  {
    v13 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v13 = 0;
    if ( v11 != 85 )
    {
      v13 = 64;
      if ( v11 != 34 )
LABEL_51:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v14 = sub_BD2BC0((__int64)a2);
  v56 = v15 + v14;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v56 >> 4) )
LABEL_49:
      BUG();
LABEL_10:
    v18 = 0;
    goto LABEL_11;
  }
  v10 = sub_BD2BC0((__int64)a2);
  if ( !(unsigned int)((v56 - v10) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_49;
  v57 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v16 = sub_BD2BC0((__int64)a2);
  v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v57);
LABEL_11:
  v19 = *((_DWORD *)a2 + 1);
  v20 = v63;
  v64 = v66;
  v61 = v63;
  v62 = 0x800000000LL;
  v21 = v12 * ((32LL * (v19 & 0x7FFFFFF) - 32 - v13 - v18) >> 5);
  v65 = 0x800000000LL;
  v22 = 0;
  v23 = 0;
  v24 = v21 - 1;
  if ( v21 != 1 )
  {
    while ( 1 )
    {
      *(_DWORD *)&v20[4 * v23] = v22;
      v26 = (unsigned int)v65;
      v27 = v22 + 1;
      LODWORD(v62) = v62 + 1;
      v28 = (unsigned int)v65 + 1LL;
      if ( v28 > HIDWORD(v65) )
      {
        sub_C8D5F0((__int64)&v64, v66, v28, 4u, v10, v27);
        v26 = (unsigned int)v65;
        LODWORD(v27) = v22 + 1;
      }
      v22 += 2;
      *(_DWORD *)&v64[4 * v26] = v27;
      LODWORD(v65) = v65 + 1;
      if ( v22 >= v24 )
        break;
      v23 = (unsigned int)v62;
      v25 = (unsigned int)v62 + 1LL;
      if ( v25 > HIDWORD(v62) )
      {
        sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 4u, v10, v25);
        v23 = (unsigned int)v62;
      }
      v20 = v61;
    }
  }
  v29 = sub_24723A0((__int64)a1, (__int64)a2, 0);
  v60 = 257;
  v30 = sub_24633A0((__int64 *)v67, 0x31u, v29, v52, (__int64)v59, 0, v58, 0);
  v31 = *a2;
  v32 = v30;
  if ( v31 == 40 )
  {
    v33 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v33 = 0;
    if ( v31 != 85 )
    {
      v33 = 64;
      if ( v31 != 34 )
        goto LABEL_51;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_33;
  v34 = sub_BD2BC0((__int64)a2);
  v49 = v35 + v34;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v49 >> 4) )
LABEL_47:
      BUG();
LABEL_33:
    v38 = 0;
    goto LABEL_34;
  }
  if ( !(unsigned int)((v49 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_33;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_47;
  v50 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v36 = sub_BD2BC0((__int64)a2);
  v38 = 32LL * (unsigned int)(*(_DWORD *)(v36 + v37 - 4) - v50);
LABEL_34:
  if ( (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v33 - v38) >> 5) == 2 )
  {
    v48 = sub_24723A0((__int64)a1, (__int64)a2, 1u);
    v60 = 257;
    v55 = (_BYTE *)sub_24633A0((__int64 *)v67, 0x31u, v48, v52, (__int64)v59, 0, v58, 0);
    v60 = 257;
    v42 = (_BYTE *)sub_A83CB0(v67, (_BYTE *)v32, v55, (__int64)v61, (unsigned int)v62, (__int64)v59);
    v60 = 257;
    v44 = (_BYTE *)sub_A83CB0(v67, (_BYTE *)v32, v55, (__int64)v64, (unsigned int)v65, (__int64)v59);
  }
  else
  {
    v39 = (unsigned int)v62;
    v60 = 257;
    v53 = (__int64)v61;
    v40 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v32 + 8));
    v41 = sub_A83CB0(v67, (_BYTE *)v32, v40, v53, v39, (__int64)v59);
    v60 = 257;
    v42 = (_BYTE *)v41;
    v51 = (unsigned int)v65;
    v54 = (__int64)v64;
    v43 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v32 + 8));
    v44 = (_BYTE *)sub_A83CB0(v67, (_BYTE *)v32, v43, v54, v51, (__int64)v59);
  }
  v60 = 257;
  v45 = sub_A82480(v67, v42, v44, (__int64)v59);
  v46 = sub_2463540(a1, *((_QWORD *)a2 + 1));
  v47 = sub_2464970(a1, v67, v45, (__int64)v46, 0);
  sub_246EF60((__int64)a1, (__int64)a2, v47);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, (__int64)a2);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  sub_F94A20(v67, (__int64)a2);
}
