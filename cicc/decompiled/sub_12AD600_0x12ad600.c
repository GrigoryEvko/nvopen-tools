// Function: sub_12AD600
// Address: 0x12ad600
//
__int64 __fastcall sub_12AD600(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4, int a5, __int64 a6)
{
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r15
  char *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rdi
  unsigned __int64 *v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  _BYTE *v31; // rsi
  _BYTE *v32; // rdi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r9
  __int64 *v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // [rsp+0h] [rbp-100h]
  __int64 v44; // [rsp+10h] [rbp-F0h]
  __int64 v45; // [rsp+10h] [rbp-F0h]
  __int64 v46; // [rsp+10h] [rbp-F0h]
  __int64 v47; // [rsp+10h] [rbp-F0h]
  __int64 v48; // [rsp+10h] [rbp-F0h]
  char *v50; // [rsp+28h] [rbp-D8h]
  __int64 v51; // [rsp+38h] [rbp-C8h]
  char *v52; // [rsp+38h] [rbp-C8h]
  __int64 v53; // [rsp+38h] [rbp-C8h]
  __int64 v54; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v55; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v56; // [rsp+58h] [rbp-A8h] BYREF
  _QWORD v57[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v58; // [rsp+70h] [rbp-90h]
  _BYTE v59[16]; // [rsp+80h] [rbp-80h] BYREF
  __int16 v60; // [rsp+90h] [rbp-70h]
  _BYTE *v61; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-58h]
  _BYTE v63[80]; // [rsp+B0h] [rbp-50h] BYREF

  v8 = (16 * a5) | 0xB0005;
  v9 = *(_QWORD *)(a6 + 16);
  v10 = a2[4];
  v11 = *(_QWORD *)(v9 + 16);
  v61 = v63;
  v12 = *(_QWORD *)(v11 + 16);
  v62 = 0x300000000LL;
  v51 = v9;
  sub_12A8750(v10, a3, &v54, &v55);
  v52 = sub_128F980((__int64)a2, v51);
  v13 = sub_128F980((__int64)a2, v11);
  v50 = sub_128F980((__int64)a2, v12);
  v14 = sub_1643350(a2[5]);
  v15 = sub_159C470(v14, v8 & 0xFF00FF, 0);
  v16 = (__int64)v52;
  v17 = v15;
  v18 = (unsigned int)v62;
  if ( (unsigned int)v62 >= HIDWORD(v62) )
  {
    sub_16CD150(&v61, v63, 0, 8);
    v18 = (unsigned int)v62;
    v16 = (__int64)v52;
  }
  *(_QWORD *)&v61[8 * v18] = v17;
  v58 = 257;
  v19 = (unsigned int)(v62 + 1);
  LODWORD(v62) = v62 + 1;
  if ( v55 == *(_QWORD *)v16 )
  {
LABEL_6:
    if ( HIDWORD(v62) > (unsigned int)v19 )
      goto LABEL_7;
    goto LABEL_29;
  }
  if ( *(_BYTE *)(v16 + 16) <= 0x10u )
  {
    v16 = sub_15A4A70(v16, v55);
    v19 = (unsigned int)v62;
    goto LABEL_6;
  }
  v60 = 257;
  v34 = sub_15FDFF0(v16, v55, v59, 0);
  v35 = a2[7];
  v36 = v34;
  if ( v35 )
  {
    v37 = (__int64 *)a2[8];
    v53 = v34;
    sub_157E9D0(v35 + 40, v34);
    v36 = v53;
    v38 = *v37;
    v39 = *(_QWORD *)(v53 + 24);
    *(_QWORD *)(v53 + 32) = v37;
    v38 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v53 + 24) = v38 | v39 & 7;
    *(_QWORD *)(v38 + 8) = v53 + 24;
    *v37 = *v37 & 7 | (v53 + 24);
  }
  v45 = v36;
  sub_164B780(v36, v57);
  v40 = a2[6];
  v16 = v45;
  if ( v40 )
  {
    v56 = a2[6];
    sub_1623A60(&v56, v40, 2);
    v16 = v45;
    v41 = v45 + 48;
    if ( *(_QWORD *)(v45 + 48) )
    {
      v43 = v45;
      v46 = v45 + 48;
      sub_161E7C0(v46);
      v16 = v43;
      v41 = v46;
    }
    v42 = v56;
    *(_QWORD *)(v16 + 48) = v56;
    if ( v42 )
    {
      v47 = v16;
      sub_1623210(&v56, v42, v41);
      v16 = v47;
    }
  }
  v19 = (unsigned int)v62;
  if ( HIDWORD(v62) <= (unsigned int)v62 )
  {
LABEL_29:
    v48 = v16;
    sub_16CD150(&v61, v63, 0, 8);
    v19 = (unsigned int)v62;
    v16 = v48;
  }
LABEL_7:
  *(_QWORD *)&v61[8 * v19] = v16;
  LODWORD(v62) = v62 + 1;
  if ( *(_BYTE *)(v54 + 8) == 16 )
  {
    v21 = (_QWORD *)sub_12A8C20(a2, v54, v13);
    v28 = (unsigned int)v62;
    if ( (unsigned int)v62 < HIDWORD(v62) )
      goto LABEL_18;
    goto LABEL_31;
  }
  v44 = v54;
  v60 = 257;
  v20 = sub_1648A60(64, 1);
  v21 = (_QWORD *)v20;
  if ( v20 )
    sub_15F9210(v20, v44, v13, 0, 0, 0);
  v22 = a2[7];
  if ( v22 )
  {
    v23 = (unsigned __int64 *)a2[8];
    sub_157E9D0(v22 + 40, v21);
    v24 = v21[3];
    v25 = *v23;
    v21[4] = v23;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    v21[3] = v25 | v24 & 7;
    *(_QWORD *)(v25 + 8) = v21 + 3;
    *v23 = *v23 & 7 | (unsigned __int64)(v21 + 3);
  }
  sub_164B780(v21, v59);
  v26 = a2[6];
  if ( v26 )
  {
    v57[0] = a2[6];
    sub_1623A60(v57, v26, 2);
    if ( v21[6] )
      sub_161E7C0(v21 + 6);
    v27 = v57[0];
    v21[6] = v57[0];
    if ( v27 )
      sub_1623210(v57, v27, v21 + 6);
  }
  v28 = (unsigned int)v62;
  if ( (unsigned int)v62 >= HIDWORD(v62) )
  {
LABEL_31:
    sub_16CD150(&v61, v63, 0, 8);
    v28 = (unsigned int)v62;
  }
LABEL_18:
  *(_QWORD *)&v61[8 * v28] = v21;
  v29 = (_QWORD *)a2[4];
  LODWORD(v62) = v62 + 1;
  v57[0] = v55;
  v30 = sub_126A190(v29, a4, (__int64)v57, 1u);
  v60 = 257;
  v31 = (_BYTE *)sub_1285290(a2 + 6, *(_QWORD *)(v30 + 24), v30, (int)v61, (unsigned int)v62, (__int64)v59, 0);
  sub_12A9060(a2, v31, v50);
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 12) &= ~1u;
  v32 = v61;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v32 != v63 )
    _libc_free(v32, v31);
  return a1;
}
