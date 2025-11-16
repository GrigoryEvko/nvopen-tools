// Function: sub_DF1EA0
// Address: 0xdf1ea0
//
__int64 __fastcall sub_DF1EA0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        unsigned __int8 a6,
        char a7,
        unsigned __int8 a8)
{
  __int64 v9; // rdi
  bool v11; // al
  __int64 v12; // rcx
  bool v13; // zf
  unsigned __int8 v14; // al
  __int64 v15; // rdi
  bool v16; // al
  char v17; // al
  __int64 v18; // r11
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 *v22; // rdx
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char **v29; // rsi
  __int64 *v30; // rdx
  _BYTE *v31; // rdi
  __int64 v32; // r15
  __int64 v33; // rbx
  __int64 v34; // rbx
  bool v35; // r14
  __int64 v36; // rbx
  __int64 v37; // rbx
  __int64 v38; // rbx
  __int64 v39; // rbx
  __int64 v40; // r9
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rcx
  char *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // [rsp-8h] [rbp-188h]
  __int64 v47; // [rsp+0h] [rbp-180h]
  _BYTE *v48; // [rsp+8h] [rbp-178h]
  char v49; // [rsp+13h] [rbp-16Dh]
  char v50; // [rsp+14h] [rbp-16Ch]
  _BYTE *v51; // [rsp+18h] [rbp-168h]
  __int64 v52; // [rsp+20h] [rbp-160h]
  __int64 v53; // [rsp+20h] [rbp-160h]
  __int64 v54; // [rsp+20h] [rbp-160h]
  __int64 v56; // [rsp+28h] [rbp-158h]
  __int64 v57; // [rsp+28h] [rbp-158h]
  __int64 v60; // [rsp+38h] [rbp-148h]
  __int64 *v61[4]; // [rsp+40h] [rbp-140h] BYREF
  __int64 v62; // [rsp+60h] [rbp-120h] BYREF
  __int64 v63; // [rsp+68h] [rbp-118h]
  __int64 v64; // [rsp+70h] [rbp-110h]
  __int64 *v65; // [rsp+80h] [rbp-100h]
  unsigned int v66; // [rsp+88h] [rbp-F8h]
  char v67; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v68; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v69; // [rsp+B8h] [rbp-C8h]
  __int64 v70; // [rsp+C0h] [rbp-C0h]
  __int64 *v71; // [rsp+D0h] [rbp-B0h]
  unsigned int v72; // [rsp+D8h] [rbp-A8h]
  char v73; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v74[3]; // [rsp+100h] [rbp-80h] BYREF
  char v75; // [rsp+118h] [rbp-68h]
  char *v76; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v77; // [rsp+128h] [rbp-58h]
  char v78; // [rsp+130h] [rbp-50h] BYREF

  if ( (unsigned __int8)*a5 <= 0x1Cu )
    goto LABEL_21;
  v9 = *((_QWORD *)a5 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  v11 = sub_BCAC40(v9, 1);
  v12 = a4;
  v13 = !v11;
  v49 = v11;
  v14 = *a5;
  if ( v13 )
    goto LABEL_13;
  if ( v14 == 57 )
  {
    if ( (a5[7] & 0x40) != 0 )
      v22 = (__int64 *)*((_QWORD *)a5 - 1);
    else
      v22 = (__int64 *)&a5[-32 * (*((_DWORD *)a5 + 1) & 0x7FFFFFF)];
    v18 = *v22;
    if ( *v22 )
    {
      v48 = (_BYTE *)v22[4];
      if ( v48 )
      {
        v47 = 1;
        v17 = a6;
        goto LABEL_28;
      }
    }
LABEL_14:
    v15 = *((_QWORD *)a5 + 1);
    goto LABEL_15;
  }
  if ( v14 != 86 )
  {
LABEL_13:
    if ( v14 <= 0x1Cu )
      goto LABEL_21;
    goto LABEL_14;
  }
  v15 = *((_QWORD *)a5 + 1);
  v56 = *((_QWORD *)a5 - 12);
  if ( *(_QWORD *)(v56 + 8) == v15 && **((_BYTE **)a5 - 4) <= 0x15u )
  {
    v52 = v12;
    v48 = (_BYTE *)*((_QWORD *)a5 - 8);
    v16 = sub_AC30F0(*((_QWORD *)a5 - 4));
    v12 = v52;
    v49 = v16;
    if ( v16 && v48 )
    {
      v17 = a6;
      v18 = v56;
      v47 = 1;
      goto LABEL_28;
    }
    v14 = *a5;
    goto LABEL_13;
  }
LABEL_15:
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  v57 = v12;
  if ( !sub_BCAC40(v15, 1) )
    goto LABEL_21;
  v12 = v57;
  if ( *a5 == 58 )
  {
    if ( (a5[7] & 0x40) != 0 )
      v30 = (__int64 *)*((_QWORD *)a5 - 1);
    else
      v30 = (__int64 *)&a5[-32 * (*((_DWORD *)a5 + 1) & 0x7FFFFFF)];
    v18 = *v30;
    if ( *v30 )
    {
      v48 = (_BYTE *)v30[4];
      if ( v48 )
      {
        v49 = 0;
        v47 = 0;
        v17 = a6 ^ 1;
        goto LABEL_28;
      }
    }
LABEL_21:
    *(_BYTE *)(a1 + 80) = 0;
    return a1;
  }
  if ( *a5 != 86 )
    goto LABEL_21;
  v53 = *((_QWORD *)a5 - 12);
  if ( *(_QWORD *)(v53 + 8) != *((_QWORD *)a5 + 1) )
    goto LABEL_21;
  v31 = (_BYTE *)*((_QWORD *)a5 - 8);
  if ( *v31 > 0x15u )
    goto LABEL_21;
  v48 = (_BYTE *)*((_QWORD *)a5 - 4);
  if ( !sub_AD7A80(v31, 1, v19, v57, v20) || !v48 )
    goto LABEL_21;
  v49 = 0;
  v12 = v57;
  v18 = v53;
  v47 = 0;
  v17 = a6 ^ 1;
LABEL_28:
  v23 = &v68;
  v51 = (_BYTE *)v18;
  v50 = v17 & a7;
  v54 = v12;
  sub_DB8AC0((__int64)&v62, (int)a2, a3, v12, v18, a6, v17 & a7, a8);
  sub_DB8AC0((__int64)&v68, (int)a2, a3, v54, (__int64)v48, a6, v50, a8);
  v24 = sub_AD64C0(*((_QWORD *)a5 + 1), v47, 0);
  if ( *v48 == 17 )
  {
    if ( (_BYTE *)v24 == v48 )
      v23 = &v62;
  }
  else
  {
    if ( *v51 != 17 )
    {
      v32 = sub_D970F0((__int64)a2);
      v60 = sub_D970F0((__int64)a2);
      v33 = sub_D970F0((__int64)a2);
      if ( v49 == a6 )
      {
        if ( v62 == v68 )
          v32 = v68;
      }
      else
      {
        v34 = v62;
        v35 = (unsigned __int8)(*a5 - 42) > 0x11u;
        if ( v34 != sub_D970F0((__int64)a2) )
        {
          v36 = v68;
          if ( v36 != sub_D970F0((__int64)a2) )
            v32 = (__int64)sub_DCF070(a2, v62, v68, v35);
        }
        v37 = v63;
        if ( v37 == sub_D970F0((__int64)a2) )
        {
          v60 = v69;
        }
        else
        {
          v38 = v69;
          if ( v38 == sub_D970F0((__int64)a2) )
            v60 = v63;
          else
            v60 = (__int64)sub_DCF070(a2, v63, v69, 0);
        }
        v39 = v64;
        v13 = v39 == sub_D970F0((__int64)a2);
        v33 = v70;
        if ( !v13 )
        {
          if ( v33 == sub_D970F0((__int64)a2) )
            v33 = v64;
          else
            v33 = (__int64)sub_DCF070(a2, v64, v70, v35);
        }
      }
      if ( sub_D96A50(v60) && !sub_D96A50(v32) )
      {
        v45 = sub_DBB9F0((__int64)a2, v32, 0, 0);
        sub_AB0910((__int64)v74, v45);
        v60 = (__int64)sub_DA26C0(a2, (__int64)v74);
        sub_969240(v74);
      }
      if ( sub_D96A50(v33) )
      {
        v33 = v60;
        if ( !sub_D96A50(v32) )
          v33 = v32;
      }
      v29 = (char **)v32;
      v61[0] = v65;
      v61[1] = (__int64 *)v66;
      v61[2] = v71;
      v61[3] = (__int64 *)v72;
      sub_D97D90((__int64)v74, v32, v60, v33, 0, v40, v61, 2);
      v43 = v77;
      *(_QWORD *)a1 = v74[0];
      *(_QWORD *)(a1 + 8) = v74[1];
      *(_QWORD *)(a1 + 16) = v74[2];
      *(_BYTE *)(a1 + 24) = v75;
      *(_QWORD *)(a1 + 32) = a1 + 48;
      *(_QWORD *)(a1 + 40) = 0x400000000LL;
      if ( (_DWORD)v43 )
      {
        v29 = &v76;
        sub_D91460(a1 + 32, &v76, v46, v43, v41, v42);
      }
      v44 = v76;
      *(_BYTE *)(a1 + 80) = 1;
      if ( v44 != &v78 )
        _libc_free(v44, v29);
      goto LABEL_35;
    }
    if ( (_BYTE *)v24 != v51 )
      v23 = &v62;
  }
  *(_QWORD *)a1 = *v23;
  *(_QWORD *)(a1 + 8) = v23[1];
  *(_QWORD *)(a1 + 16) = v23[2];
  *(_BYTE *)(a1 + 24) = *((_BYTE *)v23 + 24);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x400000000LL;
  v29 = (char **)*((unsigned int *)v23 + 10);
  if ( (_DWORD)v29 )
  {
    v29 = (char **)(v23 + 4);
    sub_D915C0(a1 + 32, (__int64)(v23 + 4), v25, v26, v27, v28);
  }
  *(_BYTE *)(a1 + 80) = 1;
LABEL_35:
  if ( v71 != (__int64 *)&v73 )
    _libc_free(v71, v29);
  if ( v65 != (__int64 *)&v67 )
    _libc_free(v65, v29);
  return a1;
}
