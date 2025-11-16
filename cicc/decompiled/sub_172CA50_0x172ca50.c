// Function: sub_172CA50
// Address: 0x172ca50
//
_QWORD *__fastcall sub_172CA50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  char v5; // al
  __int64 v6; // rdx
  char v7; // cl
  __int64 v8; // r12
  char v9; // al
  __int64 v10; // rdx
  char v11; // cl
  __int64 v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r8d
  __int64 v25; // rdx
  int v26; // r9d
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  int v31; // r9d
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rax
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rbx
  int v44; // r8d
  int v45; // r9d
  __int64 v46; // rax
  __int64 *v47; // r13
  __int64 v48; // rbx
  int v49; // r8d
  unsigned int v50; // esi
  __int64 v51; // r14
  _QWORD *result; // rax
  int v53; // r8d
  __int64 v54; // rax
  int v55; // [rsp-F8h] [rbp-F8h]
  int v56; // [rsp-F0h] [rbp-F0h]
  __int64 v57; // [rsp-E0h] [rbp-E0h]
  __int64 v58; // [rsp-D0h] [rbp-D0h]
  __int64 v59; // [rsp-D0h] [rbp-D0h]
  __int64 v60; // [rsp-C8h] [rbp-C8h]
  int v61; // [rsp-C8h] [rbp-C8h]
  __int64 v62; // [rsp-C8h] [rbp-C8h]
  __int64 v63; // [rsp-C8h] [rbp-C8h]
  __int64 v64; // [rsp-C0h] [rbp-C0h]
  __int64 v65; // [rsp-C0h] [rbp-C0h]
  _QWORD *v66; // [rsp-C0h] [rbp-C0h]
  unsigned int v67; // [rsp-B4h] [rbp-B4h] BYREF
  __int64 *v68; // [rsp-B0h] [rbp-B0h] BYREF
  __int64 v69[2]; // [rsp-A8h] [rbp-A8h] BYREF
  char v70; // [rsp-98h] [rbp-98h]
  char v71; // [rsp-97h] [rbp-97h]
  __int64 v72; // [rsp-88h] [rbp-88h]
  __int64 v73; // [rsp-80h] [rbp-80h]
  __int64 v74; // [rsp-78h] [rbp-78h]
  __int64 v75; // [rsp-70h] [rbp-70h]
  __int64 *v76; // [rsp-68h] [rbp-68h] BYREF
  __int64 v77; // [rsp-60h] [rbp-60h]
  _QWORD v78[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_BYTE *)(a2 + 16) != 51 )
    return 0;
  if ( !sub_1642F90(*(_QWORD *)a2, 32) )
    return 0;
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_BYTE *)(v4 + 16);
  if ( (unsigned __int8)(v5 - 35) > 0x11u )
    return 0;
  v6 = *(_QWORD *)(a2 - 24);
  v7 = *(_BYTE *)(v6 + 16);
  if ( (unsigned __int8)(v7 - 35) > 0x11u )
    return 0;
  if ( v5 == 51 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    v4 = *(_QWORD *)(a2 - 24);
  }
  else if ( v7 != 51 )
  {
    return 0;
  }
  v8 = *(_QWORD *)(v6 - 48);
  v9 = *(_BYTE *)(v8 + 16);
  if ( (unsigned __int8)(v9 - 35) > 0x11u )
    return 0;
  v10 = *(_QWORD *)(v6 - 24);
  v11 = *(_BYTE *)(v10 + 16);
  if ( (unsigned __int8)(v11 - 35) > 0x11u )
    return 0;
  if ( v9 == 51 )
  {
    v54 = v10;
    v10 = v8;
    v8 = v54;
    goto LABEL_11;
  }
  if ( v11 != 51 )
    return 0;
LABEL_11:
  v64 = *(_QWORD *)(v10 - 48);
  if ( (unsigned __int8)(*(_BYTE *)(v64 + 16) - 35) > 0x11u )
    return 0;
  v58 = *(_QWORD *)(v10 - 24);
  if ( (unsigned __int8)(*(_BYTE *)(v58 + 16) - 35) > 0x11u )
    return 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  if ( !(unsigned __int8)sub_1728300(v4, &v67, &v68) )
    return 0;
  *(&v72 + v67) = (__int64)v68;
  if ( !(unsigned __int8)sub_1728300(v8, &v67, &v68) )
    return 0;
  *(&v72 + v67) = (__int64)v68;
  if ( !(unsigned __int8)sub_1728300(v64, &v67, &v68) )
    return 0;
  *(&v72 + v67) = (__int64)v68;
  if ( !(unsigned __int8)sub_1728300(v58, &v67, &v68) )
    return 0;
  *(&v72 + v67) = (__int64)v68;
  v65 = v72;
  if ( !v72 )
    return 0;
  v12 = v73;
  if ( !v73 )
    return 0;
  v60 = v74;
  if ( !v74 )
    return 0;
  v59 = v75;
  if ( !v75 )
    return 0;
  v13 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) + 40LL), 4227, 0, 0);
  v78[1] = v12;
  v14 = (_QWORD *)v13;
  v76 = v78;
  v78[0] = v65;
  v77 = 0x300000002LL;
  v15 = (_QWORD *)sub_16498A0(a2);
  v16 = sub_1643350(v15);
  v19 = sub_159C470(v16, 64, 0);
  v20 = (unsigned int)v77;
  if ( (unsigned int)v77 >= HIDWORD(v77) )
  {
    sub_16CD150((__int64)&v76, v78, 0, 8, v17, v18);
    v20 = (unsigned int)v77;
  }
  v76[v20] = v19;
  v21 = *(_QWORD *)(a1 + 8);
  v71 = 1;
  v69[0] = (__int64)"prmtCall";
  v70 = 3;
  v22 = v14[3];
  LODWORD(v77) = v77 + 1;
  v23 = sub_172C570(v21, v22, (__int64)v14, v76, (unsigned int)v77, v69, 0);
  v24 = v55;
  v25 = 0;
  LODWORD(v77) = 0;
  v26 = v56;
  if ( !HIDWORD(v77) )
  {
    v57 = v23;
    sub_16CD150((__int64)&v76, v78, 0, 8, v55, v56);
    v23 = v57;
    v25 = (unsigned int)v77;
  }
  v76[v25] = v23;
  v27 = (unsigned int)(v77 + 1);
  LODWORD(v77) = v27;
  if ( HIDWORD(v77) <= (unsigned int)v27 )
  {
    sub_16CD150((__int64)&v76, v78, 0, 8, v24, v26);
    v27 = (unsigned int)v77;
  }
  v76[v27] = v60;
  LODWORD(v77) = v77 + 1;
  v28 = (_QWORD *)sub_16498A0(a2);
  v29 = sub_1643350(v28);
  v30 = sub_159C470(v29, 1040, 0);
  v32 = (unsigned int)v77;
  if ( (unsigned int)v77 >= HIDWORD(v77) )
  {
    v63 = v30;
    sub_16CD150((__int64)&v76, v78, 0, 8, v30, v31);
    v32 = (unsigned int)v77;
    v30 = v63;
  }
  v76[v32] = v30;
  v33 = *(_QWORD *)(a1 + 8);
  v71 = 1;
  v69[0] = (__int64)"prmtCall";
  v70 = 3;
  v34 = v14[3];
  LODWORD(v77) = v77 + 1;
  v35 = sub_172C570(v33, v34, (__int64)v14, v76, (unsigned int)v77, v69, 0);
  LODWORD(v77) = 0;
  v38 = v35;
  v39 = 0;
  if ( !HIDWORD(v77) )
  {
    sub_16CD150((__int64)&v76, v78, 0, 8, v36, v37);
    v39 = (unsigned int)v77;
  }
  v76[v39] = v38;
  v40 = (unsigned int)(v77 + 1);
  LODWORD(v77) = v40;
  if ( HIDWORD(v77) <= (unsigned int)v40 )
  {
    sub_16CD150((__int64)&v76, v78, 0, 8, v36, v37);
    v40 = (unsigned int)v77;
  }
  v76[v40] = v59;
  LODWORD(v77) = v77 + 1;
  v41 = (_QWORD *)sub_16498A0(a2);
  v42 = sub_1643350(v41);
  v43 = sub_159C470(v42, 16912, 0);
  v46 = (unsigned int)v77;
  if ( (unsigned int)v77 >= HIDWORD(v77) )
  {
    sub_16CD150((__int64)&v76, v78, 0, 8, v44, v45);
    v46 = (unsigned int)v77;
  }
  v76[v46] = v43;
  v71 = 1;
  v47 = v76;
  v48 = (unsigned int)(v77 + 1);
  v69[0] = (__int64)"prmtCall";
  v49 = v77 + 2;
  v70 = 3;
  v50 = v77 + 2;
  LODWORD(v77) = v77 + 1;
  v61 = v49;
  v51 = *(_QWORD *)(*v14 + 24LL);
  result = sub_1648AB0(72, v50, 0);
  if ( result )
  {
    v53 = v61;
    v62 = (__int64)result;
    sub_15F1EA0((__int64)result, **(_QWORD **)(v51 + 16), 54, (__int64)&result[-3 * v48 - 3], v53, 0);
    *(_QWORD *)(v62 + 56) = 0;
    sub_15F5B40(v62, v51, (__int64)v14, v47, v48, (__int64)v69, 0, 0);
    result = (_QWORD *)v62;
  }
  if ( v76 != v78 )
  {
    v66 = result;
    _libc_free((unsigned __int64)v76);
    return v66;
  }
  return result;
}
