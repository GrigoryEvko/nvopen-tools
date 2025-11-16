// Function: sub_DF25E0
// Address: 0xdf25e0
//
__int64 __fastcall sub_DF25E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        unsigned __int8 a6,
        char a7,
        unsigned __int8 a8)
{
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int8 v14; // r11
  char v15; // al
  _QWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned int v28; // r11d
  bool v29; // al
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rbx
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rdi
  _BYTE *v37; // rdx
  unsigned int v38; // eax
  unsigned __int8 v39; // r11
  __int64 *v40; // rbx
  bool v41; // al
  unsigned __int8 v42; // r11
  __int64 **v43; // rax
  bool v44; // al
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rdx
  _BYTE *v50; // rax
  unsigned int v51; // eax
  unsigned __int8 v52; // r11
  _QWORD *v53; // rax
  __int64 *v54; // rax
  __int64 v55; // [rsp-10h] [rbp-130h]
  __int64 v56; // [rsp-10h] [rbp-130h]
  __int64 v57; // [rsp-8h] [rbp-128h]
  __int64 v58; // [rsp-8h] [rbp-128h]
  unsigned __int8 v59; // [rsp+10h] [rbp-110h]
  unsigned __int8 v61; // [rsp+2Ch] [rbp-F4h]
  unsigned __int8 v62; // [rsp+2Ch] [rbp-F4h]
  unsigned __int8 v63; // [rsp+2Ch] [rbp-F4h]
  __int64 v65; // [rsp+30h] [rbp-F0h]
  int v66; // [rsp+38h] [rbp-E8h]
  unsigned int v67; // [rsp+4Ch] [rbp-D4h] BYREF
  __int64 v68; // [rsp+50h] [rbp-D0h] BYREF
  int v69; // [rsp+58h] [rbp-C8h]
  __int64 v70; // [rsp+60h] [rbp-C0h] BYREF
  int v71; // [rsp+68h] [rbp-B8h]
  __int64 v72[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v73[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v74; // [rsp+90h] [rbp-90h] BYREF
  __int64 v75; // [rsp+98h] [rbp-88h]
  __int64 v76; // [rsp+A0h] [rbp-80h]
  char v77; // [rsp+A8h] [rbp-78h]
  char *v78; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v79; // [rsp+B8h] [rbp-68h]
  _BYTE v80[32]; // [rsp+C0h] [rbp-60h] BYREF
  char v81; // [rsp+E0h] [rbp-40h]

  sub_DF1EA0((__int64)&v74, (__int64 *)a2, a3, (__int64)a4, (char *)a5, a6, a7, a8);
  v14 = a6;
  if ( v81 )
  {
    v28 = v79;
    *(_QWORD *)a1 = v74;
    *(_QWORD *)(a1 + 8) = v75;
    *(_QWORD *)(a1 + 16) = v76;
    *(_BYTE *)(a1 + 24) = v77;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    *(_QWORD *)(a1 + 40) = 0x400000000LL;
    if ( v28 )
    {
      a2 = (__int64)&v78;
      sub_D915C0(a1 + 32, (__int64)&v78, v57, v11, v12, v13);
      if ( !v81 )
        return a1;
    }
    goto LABEL_14;
  }
  v15 = *(_BYTE *)a5;
  if ( *(_BYTE *)a5 <= 0x1Cu )
  {
    if ( v15 == 17 )
    {
      v16 = *(_QWORD **)(a5 + 24);
      if ( *(_DWORD *)(a5 + 32) > 0x40u )
        v16 = (_QWORD *)*v16;
      if ( (v16 == 0) == a6 )
        v17 = sub_D970F0(a2);
      else
        v17 = (__int64)sub_DA2C50(a2, *(_QWORD *)(a5 + 8), 0, 0);
      sub_D97F80(a1, v17, v18, v19, v20, v21);
      return a1;
    }
LABEL_12:
    v23 = sub_DA7CB0(a2, (__int64)a4, a5, v14);
    sub_D97F80(a1, v23, v24, v25, v26, v27);
    return a1;
  }
  if ( v15 == 82 )
  {
    sub_DEECD0((__int64)&v74, (__int64 *)a2, a4, a5, a6, a7, 0);
    v29 = sub_D96A50(v74);
    if ( a8 && v29 )
    {
      sub_DEECD0(a1, (__int64 *)a2, a4, a5, a6, a7, 1u);
    }
    else
    {
      a2 = v79;
      *(_QWORD *)a1 = v74;
      *(_QWORD *)(a1 + 8) = v75;
      *(_QWORD *)(a1 + 16) = v76;
      *(_BYTE *)(a1 + 24) = v77;
      *(_QWORD *)(a1 + 32) = a1 + 48;
      *(_QWORD *)(a1 + 40) = 0x400000000LL;
      if ( (_DWORD)a2 )
      {
        a2 = (__int64)&v78;
        sub_D91460(a1 + 32, &v78, v30, v31, v32, v55);
      }
    }
LABEL_14:
    if ( v78 != v80 )
      _libc_free(v78, a2);
    return a1;
  }
  if ( v15 != 93 )
    goto LABEL_12;
  if ( *(_DWORD *)(a5 + 80) != 1 )
    goto LABEL_12;
  if ( **(_DWORD **)(a5 + 72) != 1 )
    goto LABEL_12;
  v33 = *(_QWORD *)(a5 - 32);
  if ( *(_BYTE *)v33 != 85 )
    goto LABEL_12;
  v34 = *(_QWORD *)(v33 - 32);
  if ( !v34 || *(_BYTE *)v34 || *(_QWORD *)(v34 + 24) != *(_QWORD *)(v33 + 80) || (*(_BYTE *)(v34 + 33) & 0x20) == 0 )
    goto LABEL_12;
  v35 = *(_DWORD *)(v34 + 36);
  if ( v35 != 312 )
  {
    switch ( v35 )
    {
      case 333:
      case 339:
      case 360:
      case 369:
      case 372:
        break;
      default:
        goto LABEL_12;
    }
  }
  v36 = *(_QWORD *)(v33 + 32 * (1LL - (*(_DWORD *)(v33 + 4) & 0x7FFFFFF)));
  v37 = (_BYTE *)(v36 + 24);
  if ( *(_BYTE *)v36 != 17 )
  {
    v49 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v36 + 8) + 8LL) - 17;
    if ( (unsigned int)v49 > 1 )
      goto LABEL_12;
    if ( *(_BYTE *)v36 > 0x15u )
      goto LABEL_12;
    v50 = sub_AD7630(v36, 0, v49);
    v14 = a6;
    if ( !v50 || *v50 != 17 )
      goto LABEL_12;
    v37 = v50 + 24;
  }
  v59 = v14;
  v65 = (__int64)v37;
  v66 = sub_B5B690(v33);
  v38 = sub_B5B5E0(v33);
  sub_AB3450((__int64)v72, v38, v65, v66);
  v69 = 1;
  v68 = 0;
  v71 = 1;
  v70 = 0;
  sub_AAF830((__int64)v72, (int *)&v67, (__int64)&v68, &v70);
  v39 = v59;
  if ( !a6 )
  {
    v51 = sub_B52870(v67);
    v39 = v59;
    v67 = v51;
  }
  v61 = v39;
  v40 = sub_DD8400(a2, *(_QWORD *)(v33 - 32LL * (*(_DWORD *)(v33 + 4) & 0x7FFFFFF)));
  v41 = sub_D94970((__int64)&v70, 0);
  v42 = v61;
  if ( !v41 )
  {
    v53 = sub_DA26C0((__int64 *)a2, (__int64)&v70);
    v54 = sub_DC7ED0((__int64 *)a2, (__int64)v40, (__int64)v53, 0, 0);
    v42 = v61;
    v40 = v54;
  }
  v62 = v42;
  v43 = (__int64 **)sub_DA26C0((__int64 *)a2, (__int64)&v68);
  sub_DEE290((__int64)&v74, (__int64 *)a2, a4, v67, (__int64 **)v40, v43, a7, a8);
  v44 = sub_D96A50(v74);
  v47 = v56;
  v48 = v58;
  if ( v44 && sub_D96A50(v75) )
  {
    v52 = v62;
    if ( v78 != v80 )
    {
      _libc_free(v78, a2);
      v52 = v62;
    }
    v63 = v52;
    sub_969240(&v70);
    sub_969240(&v68);
    sub_969240(v73);
    sub_969240(v72);
    v14 = v63;
    goto LABEL_12;
  }
  *(_QWORD *)a1 = v74;
  *(_QWORD *)(a1 + 8) = v75;
  *(_QWORD *)(a1 + 16) = v76;
  *(_BYTE *)(a1 + 24) = v77;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x400000000LL;
  if ( v79 )
  {
    a2 = (__int64)&v78;
    sub_D91460(a1 + 32, &v78, v47, v48, v45, v46);
  }
  if ( v78 != v80 )
    _libc_free(v78, a2);
  sub_969240(&v70);
  sub_969240(&v68);
  sub_969240(v73);
  sub_969240(v72);
  return a1;
}
