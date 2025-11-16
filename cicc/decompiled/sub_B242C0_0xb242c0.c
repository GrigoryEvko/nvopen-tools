// Function: sub_B242C0
// Address: 0xb242c0
//
void __fastcall sub_B242C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // rsi
  __int64 *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rdx
  _BYTE *v15; // rdi
  __int64 *v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rcx
  __int64 *v20; // rcx
  char *v21; // r15
  __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // eax
  unsigned int v27; // r12d
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r9
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 *v41; // rdi
  __int64 *v42; // rcx
  _QWORD **v43; // r12
  _QWORD **v44; // rbx
  _QWORD *v45; // rdi
  _BYTE *v46; // rsi
  __int64 v47; // r14
  __int64 v48; // rsi
  __int64 v49; // r8
  __int64 v50; // r15
  __int64 i; // r10
  __int64 v52; // rax
  _QWORD *v53; // r9
  __int64 v54; // rdi
  __int64 v55; // rsi
  __int64 v56; // r9
  _QWORD *v57; // rdi
  _QWORD *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rsi
  _QWORD *v61; // rcx
  __int64 v62; // [rsp+10h] [rbp-2C0h]
  __int64 v63; // [rsp+40h] [rbp-290h]
  __int64 v64; // [rsp+40h] [rbp-290h]
  __int64 v65; // [rsp+40h] [rbp-290h]
  __int64 v67; // [rsp+58h] [rbp-278h]
  unsigned int v68; // [rsp+58h] [rbp-278h]
  unsigned int v69; // [rsp+64h] [rbp-26Ch]
  __int64 *v70; // [rsp+68h] [rbp-268h] BYREF
  __int64 v71; // [rsp+78h] [rbp-258h] BYREF
  _BYTE v72[48]; // [rsp+80h] [rbp-250h] BYREF
  _BYTE *v73; // [rsp+B0h] [rbp-220h] BYREF
  __int64 v74; // [rsp+B8h] [rbp-218h]
  _BYTE v75[64]; // [rsp+C0h] [rbp-210h] BYREF
  char *v76; // [rsp+100h] [rbp-1D0h] BYREF
  int v77; // [rsp+108h] [rbp-1C8h]
  char v78; // [rsp+110h] [rbp-1C0h] BYREF
  _BYTE *v79; // [rsp+150h] [rbp-180h] BYREF
  __int64 v80; // [rsp+158h] [rbp-178h]
  _BYTE v81[72]; // [rsp+160h] [rbp-170h] BYREF
  __int64 v82; // [rsp+1A8h] [rbp-128h] BYREF
  __int64 v83; // [rsp+1B0h] [rbp-120h]
  __int64 v84; // [rsp+1B8h] [rbp-118h] BYREF
  unsigned int v85; // [rsp+1C0h] [rbp-110h]
  _BYTE *v86; // [rsp+1F8h] [rbp-D8h] BYREF
  __int64 v87; // [rsp+200h] [rbp-D0h]
  _BYTE v88[64]; // [rsp+208h] [rbp-C8h] BYREF
  _BYTE *v89; // [rsp+248h] [rbp-88h] BYREF
  __int64 v90; // [rsp+250h] [rbp-80h]
  _BYTE v91[120]; // [rsp+258h] [rbp-78h] BYREF

  v4 = a1;
  v70 = a4;
  if ( a3 && *a4 && (v5 = sub_B192F0(a1, a3, *a4)) != 0 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    v7 = *(_DWORD *)(v5 + 44) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  if ( v7 >= *(_DWORD *)(a1 + 32) )
LABEL_83:
    BUG();
  v8 = (__int64)v70;
  v62 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v6);
  v69 = *(_DWORD *)(v62 + 16) + 1;
  if ( v69 >= *((_DWORD *)v70 + 4) )
    return;
  v82 = 0;
  v83 = 1;
  v79 = v81;
  v80 = 0x800000000LL;
  v9 = &v84;
  do
    *v9++ = -4096;
  while ( v9 != (__int64 *)&v86 );
  v89 = v91;
  v73 = v75;
  v86 = v88;
  v87 = 0x800000000LL;
  v90 = 0x800000000LL;
  v74 = 0x800000000LL;
  sub_B1AE00((__int64)&v79, v8);
  v10 = v79;
  v11 = 8LL * (unsigned int)v80;
  v12 = *(_QWORD *)&v79[v11 - 8];
  v13 = (v11 >> 3) - 1;
  v14 = ((v11 >> 3) - 2) / 2;
  if ( v13 > 0 )
  {
    while ( 1 )
    {
      v15 = &v10[8 * v14];
      v61 = &v10[8 * v13];
      if ( *(_DWORD *)(*(_QWORD *)v15 + 16LL) >= *(_DWORD *)(v12 + 16) )
        break;
      *v61 = *(_QWORD *)v15;
      v13 = v14;
      if ( v14 <= 0 )
      {
        v61 = &v10[8 * v14];
        break;
      }
      v14 = (v14 - 1) / 2;
    }
  }
  else
  {
    v61 = &v79[v11 - 8];
  }
  *v61 = v12;
  v16 = &v82;
  sub_B24170((__int64)&v76, (__int64)&v82, (__int64 *)&v70);
  v17 = (unsigned int)v80;
  if ( !(_DWORD)v80 )
    goto LABEL_50;
  do
  {
    v18 = v79;
    v19 = *(_QWORD *)v79;
    if ( v17 == 1 )
      goto LABEL_18;
    v46 = &v79[8 * v17];
    v47 = *((_QWORD *)v46 - 1);
    *((_QWORD *)v46 - 1) = v19;
    v48 = v46 - 8 - (_BYTE *)v18;
    v49 = v48 >> 3;
    v50 = (v48 >> 3) & 1;
    if ( v48 > 16 )
    {
      for ( i = 0; ; i = v52 )
      {
        v52 = 2 * (i + 1);
        v53 = &v18[2 * i + 2];
        v54 = *v53;
        if ( *(_DWORD *)(*v53 + 16LL) < *(_DWORD *)(*(v53 - 1) + 16LL) )
        {
          v53 = &v18[--v52];
          v54 = *v53;
        }
        v18[i] = v54;
        if ( v52 >= ((v48 >> 3) - 1) / 2 )
          break;
      }
      if ( v50 )
      {
LABEL_79:
        v56 = v52;
        v55 = (v52 - 1) >> 1;
LABEL_73:
        while ( 1 )
        {
          v57 = &v18[v55];
          v58 = &v18[v56];
          if ( *(_DWORD *)(*v57 + 16LL) >= *(_DWORD *)(v47 + 16) )
            goto LABEL_74;
          *v58 = *v57;
          v56 = v55;
          if ( !v55 )
          {
            *v57 = v47;
            goto LABEL_18;
          }
          v55 = (v55 - 1) / 2;
        }
      }
      v55 = (v52 - 1) >> 1;
      if ( v52 != (v49 - 2) / 2 )
      {
        v56 = v52;
        goto LABEL_73;
      }
LABEL_78:
      v59 = 2 * v52 + 2;
      v60 = v18[v59 - 1];
      v52 = v59 - 1;
      *v53 = v60;
      goto LABEL_79;
    }
    v58 = v18;
    if ( !v50 && (unsigned __int64)((v48 >> 3) - 1) <= 2 )
    {
      v53 = v18;
      v52 = 0;
      goto LABEL_78;
    }
LABEL_74:
    *v58 = v47;
LABEL_18:
    v67 = v19;
    LODWORD(v80) = v80 - 1;
    sub_B1AE00((__int64)&v86, v19);
    v20 = (__int64 *)v67;
    v68 = *(_DWORD *)(v67 + 16);
    while ( 2 )
    {
      v16 = (__int64 *)*v20;
      sub_B1D150(&v76, *v20, a2);
      v21 = &v76[8 * v77];
      if ( v76 == v21 )
        goto LABEL_44;
      v22 = v4;
      v23 = (__int64 *)v76;
      v24 = v22;
      do
      {
        while ( 1 )
        {
          v34 = *v23;
          if ( *v23 )
          {
            v25 = (unsigned int)(*(_DWORD *)(v34 + 44) + 1);
            v26 = *(_DWORD *)(v34 + 44) + 1;
          }
          else
          {
            v25 = 0;
            v26 = 0;
          }
          if ( v26 >= *(_DWORD *)(v24 + 32) )
          {
            v71 = 0;
            goto LABEL_83;
          }
          v71 = *(_QWORD *)(*(_QWORD *)(v24 + 24) + 8 * v25);
          v27 = *(_DWORD *)(v71 + 16);
          if ( v69 < v27 )
          {
            v16 = &v82;
            sub_B24170((__int64)v72, (__int64)&v82, &v71);
            if ( v72[32] )
              break;
          }
LABEL_31:
          if ( v21 == (char *)++v23 )
            goto LABEL_43;
        }
        v28 = v71;
        if ( v68 < v27 )
        {
          v29 = (unsigned int)v74;
          v30 = (unsigned int)v74 + 1LL;
          if ( v30 > HIDWORD(v74) )
          {
            v16 = (__int64 *)v75;
            v65 = v71;
            sub_C8D5F0(&v73, v75, v30, 8);
            v29 = (unsigned int)v74;
            v28 = v65;
          }
          *(_QWORD *)&v73[8 * v29] = v28;
          v31 = (unsigned int)v90;
          LODWORD(v74) = v74 + 1;
          v32 = (unsigned int)v90 + 1LL;
          v33 = v71;
          if ( v32 > HIDWORD(v90) )
          {
            v16 = (__int64 *)v91;
            v64 = v71;
            sub_C8D5F0(&v89, v91, v32, 8);
            v31 = (unsigned int)v90;
            v33 = v64;
          }
          *(_QWORD *)&v89[8 * v31] = v33;
          LODWORD(v90) = v90 + 1;
          goto LABEL_31;
        }
        v35 = (unsigned int)v80;
        v36 = (unsigned int)v80 + 1LL;
        if ( v36 > HIDWORD(v80) )
        {
          v63 = v71;
          sub_C8D5F0(&v79, v81, v36, 8);
          v35 = (unsigned int)v80;
          v28 = v63;
        }
        *(_QWORD *)&v79[8 * v35] = v28;
        v16 = (__int64 *)v79;
        LODWORD(v80) = v80 + 1;
        v37 = 8LL * (unsigned int)v80;
        v38 = *(_QWORD *)&v79[v37 - 8];
        v39 = (v37 >> 3) - 1;
        v40 = ((v37 >> 3) - 2) / 2;
        if ( v39 > 0 )
        {
          while ( 1 )
          {
            v41 = &v16[v40];
            v42 = &v16[v39];
            if ( *(_DWORD *)(*v41 + 16) >= *(_DWORD *)(v38 + 16) )
            {
              *v42 = v38;
              goto LABEL_42;
            }
            *v42 = *v41;
            v39 = v40;
            if ( v40 <= 0 )
              break;
            v40 = (v40 - 1) / 2;
          }
          *v41 = v38;
        }
        else
        {
          *(_QWORD *)&v79[v37 - 8] = v38;
        }
LABEL_42:
        ++v23;
      }
      while ( v21 != (char *)v23 );
LABEL_43:
      v21 = v76;
      v4 = v24;
LABEL_44:
      if ( v21 != &v78 )
        _libc_free(v21, v16);
      if ( (_DWORD)v74 )
      {
        v20 = *(__int64 **)&v73[8 * (unsigned int)v74 - 8];
        LODWORD(v74) = v74 - 1;
        continue;
      }
      break;
    }
    v17 = (unsigned int)v80;
  }
  while ( (_DWORD)v80 );
LABEL_50:
  v43 = (_QWORD **)v86;
  v44 = (_QWORD **)&v86[8 * (unsigned int)v87];
  if ( v86 != (_BYTE *)v44 )
  {
    do
    {
      v45 = *v43;
      v16 = (__int64 *)v62;
      ++v43;
      sub_B1AE50(v45, v62);
    }
    while ( v44 != v43 );
  }
  if ( v73 != v75 )
    _libc_free(v73, v16);
  if ( v89 != v91 )
    _libc_free(v89, v16);
  if ( v86 != v88 )
    _libc_free(v86, v16);
  if ( (v83 & 1) == 0 )
  {
    v16 = (__int64 *)(8LL * v85);
    sub_C7D6A0(v84, v16, 8);
  }
  if ( v79 != v81 )
    _libc_free(v79, v16);
}
