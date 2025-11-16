// Function: sub_15BD310
// Address: 0x15bd310
//
__int64 __fastcall sub_15BD310(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10,
        int *a11,
        int a12,
        __int64 a13,
        unsigned int a14,
        char a15)
{
  __int64 v15; // r11
  __int16 v16; // r14
  _QWORD *v18; // r12
  __int64 v20; // r10
  unsigned int v21; // eax
  __int64 v22; // r10
  int v23; // r9d
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rdx
  int v28; // edi
  __int64 v29; // r9
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 result; // rax
  __int64 v33; // r13
  char v34; // bl
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // eax
  char v38; // di
  __int64 v39; // [rsp+8h] [rbp-108h]
  int v40; // [rsp+3Ch] [rbp-D4h]
  char v41; // [rsp+42h] [rbp-CEh]
  __int64 v44; // [rsp+58h] [rbp-B8h]
  __int64 v45; // [rsp+58h] [rbp-B8h]
  int v46; // [rsp+60h] [rbp-B0h]
  int v47; // [rsp+64h] [rbp-ACh]
  __int64 v48; // [rsp+68h] [rbp-A8h]
  __int64 v49; // [rsp+70h] [rbp-A0h]
  int v50; // [rsp+78h] [rbp-98h]
  unsigned int v51; // [rsp+78h] [rbp-98h]
  __int64 v53; // [rsp+80h] [rbp-90h] BYREF
  __int64 v54; // [rsp+88h] [rbp-88h] BYREF
  __int64 v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+98h] [rbp-78h] BYREF
  __int64 v57; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v58; // [rsp+A8h] [rbp-68h] BYREF
  __int64 v59; // [rsp+B0h] [rbp-60h]
  __int64 v60; // [rsp+B8h] [rbp-58h]
  int v61; // [rsp+C0h] [rbp-50h]
  int v62; // [rsp+C4h] [rbp-4Ch]
  char v63; // [rsp+C8h] [rbp-48h]
  int v64; // [rsp+CCh] [rbp-44h] BYREF
  __int64 v65; // [rsp+D0h] [rbp-40h]

  v15 = a4;
  v16 = a2;
  v18 = a1;
  if ( a14 )
  {
LABEL_38:
    v55 = a3;
    v54 = a6;
    v56 = a7;
    v53 = v15;
    v57 = a13;
    v33 = *v18 + 752LL;
    v34 = *((_BYTE *)a11 + 4);
    if ( v34 )
      v47 = *a11;
    v35 = sub_161E980(64, 5);
    v36 = v35;
    if ( v35 )
    {
      v49 = v35;
      sub_1623D80(v35, (_DWORD)v18, 12, a14, (unsigned int)&v53, 5, 0, 0);
      v36 = v49;
      *(_DWORD *)(v49 + 24) = a5;
      *(_WORD *)(v49 + 2) = v16;
      *(_DWORD *)(v49 + 28) = a12;
      *(_BYTE *)(v49 + 56) = v34;
      *(_QWORD *)(v49 + 32) = a8;
      *(_DWORD *)(v49 + 48) = a9;
      *(_QWORD *)(v49 + 40) = a10;
      if ( v34 )
        *(_DWORD *)(v49 + 52) = v47;
    }
    return sub_15BD230(v36, a14, v33);
  }
  if ( *((_BYTE *)a11 + 4) )
  {
    v55 = a4;
    LODWORD(v53) = a2;
    v37 = *a11;
    LODWORD(v56) = a5;
    v54 = a3;
    v58 = a7;
    v57 = a6;
    v59 = a8;
    v60 = a10;
    v61 = a9;
    v63 = 1;
    v62 = v37;
  }
  else
  {
    LODWORD(v53) = a2;
    v54 = a3;
    LODWORD(v56) = a5;
    v55 = a4;
    v58 = a7;
    v57 = a6;
    v59 = a8;
    v63 = 0;
    v60 = a10;
    v61 = a9;
  }
  v20 = *a1;
  v64 = a12;
  v65 = a13;
  v48 = *(_QWORD *)(v20 + 760);
  v50 = *(_DWORD *)(v20 + 776);
  if ( !v50 )
    goto LABEL_37;
  if ( a6 != 0 && a3 != 0 && a2 == 13 && *(_BYTE *)a6 == 13 && *(_QWORD *)(a6 + 8 * (7LL - *(unsigned int *)(a6 + 8))) )
  {
    v44 = v20;
    v21 = sub_15B2D00(&v54, &v57);
    v22 = v44;
    v15 = a4;
  }
  else
  {
    v45 = v20;
    v21 = sub_15B4C20((int *)&v53, &v54, &v55, (int *)&v56, &v57, &v58, &v64);
    v15 = a4;
    v22 = v45;
  }
  v23 = v50 - 1;
  v24 = (v50 - 1) & v21;
  v51 = v24;
  v25 = (__int64 *)(v48 + 8 * v24);
  v26 = *v25;
  if ( *v25 == -8 )
  {
LABEL_37:
    result = 0;
    if ( !a15 )
      return result;
    goto LABEL_38;
  }
  v46 = 1;
  v40 = v23;
  while ( 1 )
  {
    if ( v26 == -16 )
      goto LABEL_47;
    v28 = *(unsigned __int16 *)(v26 + 2);
    if ( v54 != 0
      && (_DWORD)v53 == 13
      && v57 != 0
      && *(_BYTE *)v57 == 13
      && *(_QWORD *)(v57 + 8 * (7LL - *(unsigned int *)(v57 + 8)))
      && (_WORD)v28 == 13 )
    {
      v29 = *(unsigned int *)(v26 + 8);
      v39 = v29;
      v30 = *(_QWORD *)(v26 + 8 * (2 - v29));
      if ( v54 != v30 )
        goto LABEL_47;
      if ( v30 )
      {
        if ( v57 == *(_QWORD *)(v26 + 8 * (1 - v29)) )
          goto LABEL_35;
        goto LABEL_21;
      }
    }
    else
    {
      if ( (_DWORD)v53 != v28 )
        goto LABEL_47;
      v39 = *(unsigned int *)(v26 + 8);
      v30 = *(_QWORD *)(v26 + 8 * (2 - v39));
    }
    if ( v54 != v30 )
      goto LABEL_47;
LABEL_21:
    v31 = v26;
    if ( *(_BYTE *)v26 != 15 )
      v31 = *(_QWORD *)(v26 - 8 * v39);
    if ( v55 == v31
      && (_DWORD)v56 == *(_DWORD *)(v26 + 24)
      && v57 == *(_QWORD *)(v26 + 8 * (1 - v39))
      && v58 == *(_QWORD *)(v26 + 8 * (3 - v39))
      && v59 == *(_QWORD *)(v26 + 32)
      && v61 == *(_DWORD *)(v26 + 48)
      && v60 == *(_QWORD *)(v26 + 40) )
    {
      break;
    }
LABEL_47:
    v51 = v40 & (v51 + v46);
    v25 = (__int64 *)(v48 + 8LL * v51);
    v26 = *v25;
    if ( *v25 == -8 )
    {
      v18 = a1;
      v16 = a2;
      goto LABEL_37;
    }
    ++v46;
  }
  v41 = *(_BYTE *)(v26 + 56);
  if ( v41 )
  {
    if ( v63 )
    {
      if ( *(_DWORD *)(v26 + 52) != v62 )
        goto LABEL_47;
      goto LABEL_33;
    }
    v38 = 0;
  }
  else
  {
    v38 = v63;
  }
  if ( v41 != v38 )
    goto LABEL_47;
LABEL_33:
  if ( v64 != *(_DWORD *)(v26 + 28) || v65 != *(_QWORD *)(v26 + 8 * (4 - v39)) )
    goto LABEL_47;
LABEL_35:
  v18 = a1;
  v16 = a2;
  if ( v25 == (__int64 *)(*(_QWORD *)(v22 + 760) + 8LL * *(unsigned int *)(v22 + 776)) )
    goto LABEL_37;
  result = *v25;
  if ( !result )
    goto LABEL_37;
  return result;
}
