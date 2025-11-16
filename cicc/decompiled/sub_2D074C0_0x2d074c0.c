// Function: sub_2D074C0
// Address: 0x2d074c0
//
__int64 __fastcall sub_2D074C0(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rsi
  __int64 v3; // r8
  __int64 v4; // r9
  _BYTE *v5; // r15
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 i; // r13
  char v10; // al
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  _BYTE *v13; // r13
  unsigned __int64 v14; // r15
  __int64 *v15; // rdx
  _BYTE *v16; // r12
  unsigned __int8 v17; // al
  __int64 v18; // rdi
  __int64 v19; // r14
  _BYTE *v20; // rsi
  __int64 v21; // rax
  _QWORD *v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r8
  char *v35; // r9
  unsigned int v36; // edx
  __int64 v37; // rcx
  char *v38; // rdi
  char *v39; // rax
  _QWORD *v40; // r13
  _QWORD *v41; // rbx
  char v42; // al
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 *v46; // r12
  __int64 v47; // r15
  _QWORD *v48; // r14
  __int64 v49; // rax
  char v50; // si
  __int64 v51; // r15
  __int64 v52; // r14
  __int64 v53; // rbx
  unsigned __int8 *v54; // r13
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // r12
  __int64 v57; // rax
  unsigned __int64 v58; // r12
  int v59; // eax
  __int64 v60; // rax
  __int64 v61; // r12
  __int64 v62; // rax
  __int64 v64; // [rsp+18h] [rbp-808h]
  __int64 *v65; // [rsp+20h] [rbp-800h]
  unsigned __int8 v66; // [rsp+2Fh] [rbp-7F1h]
  __int64 v67; // [rsp+30h] [rbp-7F0h]
  __int64 v69; // [rsp+38h] [rbp-7E8h]
  __int64 *v70; // [rsp+38h] [rbp-7E8h]
  _BYTE *v71; // [rsp+40h] [rbp-7E0h] BYREF
  __int64 v72; // [rsp+48h] [rbp-7D8h]
  _BYTE v73[64]; // [rsp+50h] [rbp-7D0h] BYREF
  _BYTE *v74; // [rsp+90h] [rbp-790h] BYREF
  __int64 v75; // [rsp+98h] [rbp-788h]
  _BYTE v76[64]; // [rsp+A0h] [rbp-780h] BYREF
  __int64 *v77; // [rsp+E0h] [rbp-740h] BYREF
  __int64 v78; // [rsp+E8h] [rbp-738h]
  _BYTE v79[64]; // [rsp+F0h] [rbp-730h] BYREF
  char v80[8]; // [rsp+130h] [rbp-6F0h] BYREF
  unsigned __int64 v81; // [rsp+138h] [rbp-6E8h]
  char v82; // [rsp+14Ch] [rbp-6D4h]
  char *v83; // [rsp+190h] [rbp-690h]
  int v84; // [rsp+198h] [rbp-688h]
  char v85; // [rsp+1A0h] [rbp-680h] BYREF
  char v86[8]; // [rsp+2E0h] [rbp-540h] BYREF
  unsigned __int64 v87; // [rsp+2E8h] [rbp-538h]
  char v88; // [rsp+2FCh] [rbp-524h]
  char *v89; // [rsp+340h] [rbp-4E0h]
  unsigned int v90; // [rsp+348h] [rbp-4D8h]
  char v91; // [rsp+350h] [rbp-4D0h] BYREF
  const char *v92; // [rsp+490h] [rbp-390h] BYREF
  unsigned __int64 v93; // [rsp+498h] [rbp-388h]
  const char *v94; // [rsp+4A0h] [rbp-380h]
  char v95; // [rsp+4ACh] [rbp-374h]
  __int16 v96; // [rsp+4B0h] [rbp-370h]
  char *v97; // [rsp+4F0h] [rbp-330h]
  char v98; // [rsp+500h] [rbp-320h] BYREF
  char v99[8]; // [rsp+640h] [rbp-1E0h] BYREF
  unsigned __int64 v100; // [rsp+648h] [rbp-1D8h]
  char v101; // [rsp+65Ch] [rbp-1C4h]
  char *v102; // [rsp+6A0h] [rbp-180h]
  char v103; // [rsp+6B0h] [rbp-170h] BYREF

  v2 = *a1;
  v71 = v73;
  v72 = 0x800000000LL;
  v75 = 0x800000000LL;
  v74 = v76;
  sub_2D06F50((__int64)&v74, v2);
  if ( &v74[8 * (unsigned int)v75] != v74 )
  {
    v67 = v1;
    v5 = &v74[8 * (unsigned int)v75];
    v6 = (unsigned __int64)v74;
    do
    {
      v7 = *((_QWORD *)v5 - 1);
      v8 = *(_QWORD *)(v7 + 56);
      for ( i = v7 + 48; i != v8; v8 = *(_QWORD *)(v8 + 8) )
      {
        while ( 1 )
        {
          if ( !v8 )
            BUG();
          v10 = *(_BYTE *)(v8 - 24);
          if ( v10 == 90 || v10 == 93 )
            break;
          v8 = *(_QWORD *)(v8 + 8);
          if ( i == v8 )
            goto LABEL_12;
        }
        v11 = (unsigned int)v72;
        v12 = (unsigned int)v72 + 1LL;
        if ( v12 > HIDWORD(v72) )
        {
          sub_C8D5F0((__int64)&v71, v73, v12, 8u, v3, v4);
          v11 = (unsigned int)v72;
        }
        *(_QWORD *)&v71[8 * v11] = v8 - 24;
        LODWORD(v72) = v72 + 1;
      }
LABEL_12:
      v5 -= 8;
    }
    while ( (_BYTE *)v6 != v5 );
    v1 = v67;
  }
  v13 = &v71[8 * (unsigned int)v72];
  if ( v13 != v71 )
  {
    v66 = 0;
    v14 = (unsigned __int64)v71;
    while ( 1 )
    {
      v19 = *(_QWORD *)v14;
      if ( (*(_BYTE *)(*(_QWORD *)v14 + 7LL) & 0x40) != 0 )
        v15 = *(__int64 **)(v19 - 8);
      else
        v15 = (__int64 *)(v19 - 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF));
      v16 = (_BYTE *)*v15;
      v17 = *(_BYTE *)*v15;
      if ( v17 > 0x1Cu )
      {
        if ( *(_BYTE *)v19 != 90 || (v20 = (_BYTE *)v15[4], *v20 <= 0x1Cu) )
        {
          v18 = *((_QWORD *)v16 + 5);
          if ( v17 != 84 )
            goto LABEL_20;
          goto LABEL_28;
        }
        if ( (unsigned __int8)sub_B19DB0(a1[3], (__int64)v20, *v15) )
        {
          v18 = *((_QWORD *)v16 + 5);
          if ( *v16 != 84 )
          {
LABEL_20:
            if ( *(_QWORD *)(v19 + 40) != v18 )
            {
              sub_B44530((_QWORD *)v19, (__int64)v16);
              v66 = 1;
            }
            goto LABEL_22;
          }
LABEL_28:
          v21 = sub_AA4FF0(v18);
          v22 = (_QWORD *)v21;
          if ( !v21 )
            BUG();
          v18 = *(_QWORD *)(v21 + 16);
          v16 = (_BYTE *)(v21 - 24);
          v23 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v23 != v18 + 48 )
          {
            if ( !v23 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 <= 0xA && v16 == (_BYTE *)(v23 - 24) )
            {
              if ( *(_QWORD **)(v18 + 56) == v22 || (v24 = *v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                BUG();
              v18 = *(_QWORD *)(v24 + 16);
              v16 = (_BYTE *)(v24 - 24);
            }
          }
          goto LABEL_20;
        }
      }
LABEL_22:
      v14 += 8LL;
      if ( v13 == (_BYTE *)v14 )
        goto LABEL_37;
    }
  }
  v66 = 0;
LABEL_37:
  v78 = 0x800000000LL;
  v25 = *a1;
  v77 = (__int64 *)v79;
  sub_2D06DA0((__int64)&v92, v25);
  sub_CE3710((__int64)v80, (__int64)&v92, v26, v27, v28, v29);
  sub_CE3710((__int64)v86, (__int64)v99, v30, v31, v32, v33);
  v69 = v1;
LABEL_38:
  v36 = v84;
  while ( 1 )
  {
    v37 = 40LL * v36;
    if ( v36 == (unsigned __int64)v90 )
      break;
LABEL_43:
    v40 = (_QWORD *)(*(_QWORD *)&v83[v37 - 8] + 48LL);
    v41 = (_QWORD *)(*v40 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v40 != v41 )
    {
      do
      {
        while ( 1 )
        {
          if ( !v41 )
            BUG();
          v42 = *((_BYTE *)v41 - 24);
          if ( v42 == 94 || v42 == 91 )
            break;
          v41 = (_QWORD *)(*v41 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v40 == v41 )
            goto LABEL_52;
        }
        v43 = (unsigned int)v78;
        v44 = (unsigned int)v78 + 1LL;
        if ( v44 > HIDWORD(v78) )
        {
          sub_C8D5F0((__int64)&v77, v79, v44, 8u, v34, (__int64)v35);
          v43 = (unsigned int)v78;
        }
        v77[v43] = (__int64)(v41 - 3);
        LODWORD(v78) = v78 + 1;
        v41 = (_QWORD *)(*v41 & 0xFFFFFFFFFFFFFFF8LL);
      }
      while ( v40 != v41 );
LABEL_52:
      v36 = v84;
    }
    v84 = --v36;
    if ( v36 )
    {
      sub_CE27D0((__int64)v80);
      goto LABEL_38;
    }
  }
  v34 = (__int64)&v83[v37];
  v35 = v89;
  if ( &v83[v37] != v83 )
  {
    v38 = v89;
    v39 = v83;
    while ( *((_QWORD *)v39 + 4) == *((_QWORD *)v38 + 4)
         && *((_DWORD *)v39 + 6) == *((_DWORD *)v38 + 6)
         && *((_DWORD *)v39 + 2) == *((_DWORD *)v38 + 2) )
    {
      v39 += 40;
      v38 += 40;
      if ( (char *)v34 == v39 )
        goto LABEL_58;
    }
    goto LABEL_43;
  }
LABEL_58:
  v45 = v69;
  if ( v89 != &v91 )
    _libc_free((unsigned __int64)v89);
  if ( !v88 )
    _libc_free(v87);
  if ( v83 != &v85 )
    _libc_free((unsigned __int64)v83);
  if ( !v82 )
    _libc_free(v81);
  if ( v102 != &v103 )
    _libc_free((unsigned __int64)v102);
  if ( !v101 )
    _libc_free(v100);
  if ( v97 != &v98 )
    _libc_free((unsigned __int64)v97);
  if ( !v95 )
    _libc_free(v93);
  v46 = v77;
  v65 = &v77[(unsigned int)v78];
  if ( v65 != v77 )
  {
    v70 = v77;
    v47 = v45;
    while ( 1 )
    {
      v48 = (_QWORD *)*v70;
      if ( !*(_QWORD *)(*v70 + 16) )
        goto LABEL_102;
      v49 = v47;
      v50 = 0;
      v51 = *v70;
      v52 = *(_QWORD *)(*v70 + 16);
      v53 = v49;
      do
      {
        while ( 1 )
        {
          v56 = *(_QWORD *)(v52 + 24);
          if ( *(_BYTE *)v56 <= 0x1Cu || *(_QWORD *)(v56 + 40) == *(_QWORD *)(v51 + 40) )
            goto LABEL_79;
          if ( *(_BYTE *)v56 == 84 )
          {
            v57 = *(_QWORD *)(*(_QWORD *)(v56 - 8)
                            + 32LL * *(unsigned int *)(v56 + 72)
                            + 8LL * (unsigned int)sub_BD2910(v52));
            v58 = *(_QWORD *)(v57 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v58 == v57 + 48 )
            {
              v56 = 0;
            }
            else
            {
              if ( !v58 )
                BUG();
              v59 = *(unsigned __int8 *)(v58 - 24);
              v56 = v58 - 24;
              if ( (unsigned int)(v59 - 30) >= 0xB )
                v56 = 0;
            }
          }
          v60 = *(_QWORD *)(v51 + 16);
          v61 = v56 + 24;
          if ( v60 )
          {
            if ( !*(_QWORD *)(v60 + 8) )
              break;
          }
          LOWORD(v53) = 0;
          v54 = (unsigned __int8 *)sub_B47F80((_BYTE *)v51);
          v92 = sub_BD5D20(v51);
          v96 = 773;
          v93 = v55;
          v94 = ".pre-remat";
          sub_BD6B50(v54, &v92);
          sub_B44220(v54, v61, v53);
          v50 = 1;
LABEL_79:
          v52 = *(_QWORD *)(v52 + 8);
          if ( !v52 )
            goto LABEL_90;
        }
        v62 = v64;
        LOWORD(v62) = 0;
        v64 = v62;
        sub_B444E0((_QWORD *)v51, v61, v62);
        v52 = *(_QWORD *)(v52 + 8);
        v50 = 1;
      }
      while ( v52 );
LABEL_90:
      v48 = (_QWORD *)v51;
      v47 = v53;
      if ( v48[2] )
      {
        v66 |= v50;
        goto LABEL_92;
      }
LABEL_102:
      sub_B43D60(v48);
      v66 = 1;
LABEL_92:
      if ( v65 == ++v70 )
      {
        v46 = v77;
        break;
      }
    }
  }
  if ( v46 != (__int64 *)v79 )
    _libc_free((unsigned __int64)v46);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  return v66;
}
