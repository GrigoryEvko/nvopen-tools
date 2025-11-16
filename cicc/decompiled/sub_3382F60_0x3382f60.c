// Function: sub_3382F60
// Address: 0x3382f60
//
void __fastcall sub_3382F60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        unsigned int a9,
        int a10)
{
  _QWORD *v14; // rbx
  char v15; // dl
  __int64 v16; // rax
  __int64 v17; // rdi
  bool v18; // al
  __int64 v19; // rdi
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  _BYTE *v23; // rdi
  bool v24; // al
  __int64 v25; // r11
  __int64 v26; // r10
  int v27; // eax
  __int64 *v28; // rdx
  char v29; // al
  __int64 v30; // rsi
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // r12
  int v36; // r10d
  __int64 v37; // rcx
  __int64 v38; // rax
  int v39; // r9d
  unsigned __int8 v40; // r8
  unsigned int v41; // ecx
  bool v42; // al
  char v43; // al
  __int64 v44; // rax
  unsigned int v45; // ecx
  unsigned __int8 v46; // [rsp+8h] [rbp-88h]
  int v47; // [rsp+8h] [rbp-88h]
  int v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+10h] [rbp-80h]
  int v50; // [rsp+10h] [rbp-80h]
  int v51; // [rsp+10h] [rbp-80h]
  int v52; // [rsp+10h] [rbp-80h]
  int v53; // [rsp+10h] [rbp-80h]
  char v54; // [rsp+18h] [rbp-78h]
  int v55; // [rsp+18h] [rbp-78h]
  int v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  int v58; // [rsp+18h] [rbp-78h]
  int v59; // [rsp+20h] [rbp-70h]
  unsigned int v60; // [rsp+20h] [rbp-70h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  int v63; // [rsp+20h] [rbp-70h]
  __int64 v64; // [rsp+20h] [rbp-70h]
  __int64 v65; // [rsp+20h] [rbp-70h]
  __int64 v66; // [rsp+28h] [rbp-68h]
  __int64 v67; // [rsp+28h] [rbp-68h]
  __int64 v68; // [rsp+28h] [rbp-68h]
  __int64 v69; // [rsp+28h] [rbp-68h]
  __int64 *v70; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 *v72; // [rsp+40h] [rbp-50h] BYREF
  __int64 v73; // [rsp+48h] [rbp-48h]
  unsigned __int64 v74; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v75[14]; // [rsp+58h] [rbp-38h] BYREF

  v14 = (_QWORD *)a5;
  LODWORD(a5) = a10;
  v15 = *(_BYTE *)a2;
  v73 = (__int64)&v71;
  v16 = *(_QWORD *)(a2 + 16);
  v72 = 0;
  if ( v16 && !*(_QWORD *)(v16 + 8) && v15 == 59 )
  {
    v62 = a6;
    v29 = sub_995B10(&v72, *(_QWORD *)(a2 - 64));
    v30 = *(_QWORD *)(a2 - 32);
    a6 = v62;
    LODWORD(a5) = a10;
    if ( v29 && v30 )
    {
      *(_QWORD *)v73 = v30;
    }
    else
    {
      v43 = sub_995B10(&v72, v30);
      a6 = v62;
      LODWORD(a5) = a10;
      if ( !v43 || (v44 = *(_QWORD *)(a2 - 64)) == 0 )
      {
LABEL_32:
        v15 = *(_BYTE *)a2;
        goto LABEL_2;
      }
      *(_QWORD *)v73 = v44;
    }
    if ( *(_BYTE *)v71 <= 0x1Cu || v14[2] == *(_QWORD *)(v71 + 40) )
    {
      sub_3382F60(a1, v71, a3, a4, (_DWORD)v14, a6, a7, a8, a9, (unsigned __int8)a5 ^ 1);
      return;
    }
    goto LABEL_32;
  }
LABEL_2:
  if ( (unsigned __int8)v15 <= 0x1Cu )
    goto LABEL_23;
  v17 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
    v17 = **(_QWORD **)(v17 + 16);
  v59 = a5;
  v66 = a6;
  v18 = sub_BCAC40(v17, 1);
  a6 = v66;
  LODWORD(a5) = v59;
  if ( !v18 )
    goto LABEL_55;
  if ( *(_BYTE *)a2 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v31 = *(__int64 **)(a2 - 8);
    else
      v31 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v26 = *v31;
    if ( !*v31 )
      goto LABEL_55;
    v25 = v31[4];
    if ( !v25 )
      goto LABEL_55;
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 8);
    if ( *(_BYTE *)a2 != 86 )
      goto LABEL_9;
    v67 = *(_QWORD *)(a2 - 96);
    if ( *(_QWORD *)(v67 + 8) != v19 || **(_BYTE **)(a2 - 32) > 0x15u )
      goto LABEL_9;
    v52 = v59;
    v57 = a6;
    v64 = *(_QWORD *)(a2 - 64);
    v42 = sub_AC30F0(*(_QWORD *)(a2 - 32));
    v25 = v64;
    a6 = v57;
    LODWORD(a5) = v52;
    v26 = v67;
    if ( !v42 || !v64 )
    {
LABEL_55:
      v19 = *(_QWORD *)(a2 + 8);
LABEL_9:
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
        v19 = **(_QWORD **)(v19 + 16);
      v60 = a5;
      v68 = a6;
      v20 = sub_BCAC40(v19, 1);
      a6 = v68;
      a5 = v60;
      if ( !v20 )
        goto LABEL_23;
      if ( *(_BYTE *)a2 == 58 )
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v28 = *(__int64 **)(a2 - 8);
        else
          v28 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v26 = *v28;
        if ( !*v28 )
          goto LABEL_23;
        v25 = v28[4];
        if ( !v25 )
          goto LABEL_23;
        goto LABEL_18;
      }
      if ( *(_BYTE *)a2 == 86 )
      {
        v69 = *(_QWORD *)(a2 - 96);
        if ( *(_QWORD *)(v69 + 8) == *(_QWORD *)(a2 + 8) )
        {
          v23 = *(_BYTE **)(a2 - 64);
          if ( *v23 <= 0x15u )
          {
            v54 = v60;
            v61 = a6;
            v49 = *(_QWORD *)(a2 - 32);
            v24 = sub_AD7A80(v23, 1, v21, v22, a5);
            a6 = v61;
            LOBYTE(a5) = v54;
            if ( v24 )
            {
              v25 = v49;
              v26 = v69;
              if ( v49 )
              {
LABEL_18:
                v27 = ((_BYTE)a5 == 0) + 28;
                goto LABEL_38;
              }
            }
          }
        }
      }
LABEL_23:
      sub_3376CF0(a1, (char *)a2, a3, a4, (__int64)v14, a6, a8, a9, a5);
      return;
    }
  }
  v27 = 28 - (((_BYTE)a5 == 0) - 1);
LABEL_38:
  if ( v27 != a7 )
    goto LABEL_23;
  v32 = *(_QWORD *)(a2 + 16);
  if ( !v32 )
    goto LABEL_23;
  if ( *(_QWORD *)(v32 + 8) )
    goto LABEL_23;
  v33 = v14[2];
  if ( v33 != *(_QWORD *)(a2 + 40) || *(_BYTE *)v26 > 0x1Cu && v33 != *(_QWORD *)(v26 + 40) )
    goto LABEL_23;
  if ( *(_BYTE *)v25 > 0x1Cu && v33 != *(_QWORD *)(v25 + 40) )
    goto LABEL_23;
  v46 = a5;
  v50 = a6;
  v34 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 40LL);
  LOBYTE(v73) = 0;
  v55 = v25;
  v63 = v26;
  v35 = sub_2E7AAE0(v34, v33, (__int64)v72, 0);
  v70 = (__int64 *)v14[1];
  sub_2E33BD0(v14[4] + 320LL, v35);
  v36 = v63;
  v37 = *v70;
  v38 = *(_QWORD *)v35 & 7LL;
  *(_QWORD *)(v35 + 8) = v70;
  v39 = v50;
  v40 = v46;
  v37 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v35 = v37 | v38;
  *(_QWORD *)(v37 + 8) = v35;
  *v70 = v35 | *v70 & 7;
  if ( a7 == 29 )
  {
    v48 = v55;
    v58 = v50;
    v65 = a8 >> 1;
    v45 = 0x80000000;
    if ( (unsigned __int64)a9 + v65 <= 0x80000000 )
      v45 = (a8 >> 1) + a9;
    v53 = v40;
    sub_3382F60(a1, v36, a3, v35, (_DWORD)v14, v39, 29, a8 >> 1, v45, v40);
    v72 = &v74;
    v73 = 0x200000002LL;
    v74 = v65 | ((unsigned __int64)a9 << 32);
    sub_27DE390((unsigned int *)&v74, v75);
    sub_3382F60(a1, v48, a3, a4, v35, v58, 29, *(_DWORD *)v72, *((_DWORD *)v72 + 1), v53);
  }
  else
  {
    v47 = v55;
    v56 = v50;
    v41 = 0x80000000;
    if ( (a9 >> 1) + (unsigned __int64)a8 <= 0x80000000 )
      v41 = (a9 >> 1) + a8;
    v51 = v40;
    sub_3382F60(a1, v63, v35, a4, (_DWORD)v14, v39, 28, v41, a9 >> 1, v40);
    v72 = &v74;
    v73 = 0x200000002LL;
    v74 = a8 | ((unsigned __int64)(a9 >> 1) << 32);
    sub_27DE390((unsigned int *)&v74, v75);
    sub_3382F60(a1, v47, a3, a4, v35, v56, 28, *(_DWORD *)v72, *((_DWORD *)v72 + 1), v51);
  }
  if ( v72 != &v74 )
    _libc_free((unsigned __int64)v72);
}
