// Function: sub_25DE870
// Address: 0x25de870
//
void __fastcall sub_25DE870(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int8 *a4, _BYTE *a5, __int64 *a6)
{
  _QWORD *v8; // rax
  __int64 *v9; // rax
  __int64 **v10; // r15
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  __int16 v13; // r12
  _QWORD *v14; // rbx
  _QWORD *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  unsigned __int8 v18; // si
  __int64 v19; // rdx
  unsigned int v20; // r15d
  _QWORD *v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // r14
  const char *v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  _BYTE *v32; // r14
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  _QWORD *v35; // rbx
  __int64 v36; // r14
  char v37; // r13
  __int64 v38; // rsi
  _BYTE *v39; // r15
  __int64 i; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // r12
  __int64 v44; // rdx
  __int16 v45; // r13
  _QWORD *v46; // rax
  __int16 v47; // r13
  __int64 v48; // r11
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  unsigned __int8 *v53; // r12
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rcx
  __int64 *v58; // rax
  __int64 *v59; // rax
  char v60; // r12
  __int64 *v61; // rax
  __int64 v62; // r12
  _QWORD *v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v71; // [rsp+28h] [rbp-138h]
  _BYTE **v72; // [rsp+30h] [rbp-130h]
  __int64 v74; // [rsp+48h] [rbp-118h]
  __int16 v75; // [rsp+48h] [rbp-118h]
  __int64 v76; // [rsp+50h] [rbp-110h]
  __int64 v77; // [rsp+50h] [rbp-110h]
  __int64 v78; // [rsp+50h] [rbp-110h]
  _QWORD *v79; // [rsp+50h] [rbp-110h]
  __int64 v80; // [rsp+60h] [rbp-100h]
  _QWORD *v81; // [rsp+60h] [rbp-100h]
  char v82; // [rsp+60h] [rbp-100h]
  char v83; // [rsp+60h] [rbp-100h]
  _QWORD *v84; // [rsp+68h] [rbp-F8h]
  _BYTE **v85; // [rsp+68h] [rbp-F8h]
  _BYTE *v86; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v87; // [rsp+78h] [rbp-E8h]
  _BYTE v88[32]; // [rsp+80h] [rbp-E0h] BYREF
  const char *v89; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-B8h]
  _QWORD v91[2]; // [rsp+B0h] [rbp-B0h] BYREF
  __int16 v92; // [rsp+C0h] [rbp-A0h]
  __int64 v93; // [rsp+D0h] [rbp-90h]
  __int64 v94; // [rsp+D8h] [rbp-88h]
  __int16 v95; // [rsp+E0h] [rbp-80h]
  _QWORD *v96; // [rsp+E8h] [rbp-78h]
  void **v97; // [rsp+F0h] [rbp-70h]
  void **v98; // [rsp+F8h] [rbp-68h]
  __int64 v99; // [rsp+100h] [rbp-60h]
  int v100; // [rsp+108h] [rbp-58h]
  __int16 v101; // [rsp+10Ch] [rbp-54h]
  char v102; // [rsp+10Eh] [rbp-52h]
  __int64 v103; // [rsp+110h] [rbp-50h]
  __int64 v104; // [rsp+118h] [rbp-48h]
  void *v105; // [rsp+120h] [rbp-40h] BYREF
  void *v106; // [rsp+128h] [rbp-38h] BYREF

  v8 = (_QWORD *)sub_BD5C60(a1);
  v9 = (__int64 *)sub_BCB2B0(v8);
  v10 = (__int64 **)sub_BCD420(v9, a3);
  v80 = *(_QWORD *)(a1 + 40);
  v76 = sub_ACA8A0(v10);
  BYTE4(v86) = 0;
  v89 = sub_BD5D20(a1);
  v91[0] = ".body";
  v11 = *(_BYTE *)(a1 + 33);
  v92 = 773;
  v90 = v12;
  v13 = (v11 >> 2) & 7;
  v14 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v14 )
    sub_B30000((__int64)v14, v80, v10, 0, 7, v76, (__int64)&v89, 0, v13, (__int64)v86, 0);
  if ( (unsigned int)*a4 - 12 > 1 )
  {
    v64 = a2[4];
    if ( v64 == a2[5] + 48LL || !v64 )
      v65 = 0;
    else
      v65 = v64 - 24;
    v96 = (_QWORD *)sub_BD5C60(v65);
    v90 = 0x200000000LL;
    v101 = 512;
    v105 = &unk_49DA100;
    v97 = &v105;
    v98 = &v106;
    v95 = 0;
    v106 = &unk_49DA0B0;
    v89 = (const char *)v91;
    v99 = 0;
    v100 = 0;
    v102 = 7;
    v103 = 0;
    v104 = 0;
    v93 = 0;
    v94 = 0;
    sub_D5F1F0((__int64)&v89, v65);
    v66 = sub_BCB2E0(v96);
    v67 = sub_ACD640(v66, a3, 0);
    sub_B34240((__int64)&v89, (__int64)v14, (__int64)a4, v67, 0, 0, 0, 0, 0);
    nullsub_61();
    v105 = &unk_49DA100;
    nullsub_63();
    if ( v89 != (const char *)v91 )
      _libc_free((unsigned __int64)v89);
  }
  sub_BD84D0((__int64)a2, (__int64)v14);
  v15 = (_QWORD *)sub_BD5C60(a1);
  v81 = (_QWORD *)sub_BCB2A0(v15);
  v16 = (__int64 *)sub_BD5C60(a1);
  v77 = sub_ACD720(v16);
  v89 = sub_BD5D20(a1);
  v91[0] = ".init";
  v17 = *(_QWORD *)(a1 + 8);
  v92 = 773;
  v18 = *(_BYTE *)(a1 + 33);
  v90 = v19;
  v20 = *(_DWORD *)(v17 + 8) >> 8;
  v21 = sub_BD2C40(88, unk_3F0FAE8);
  v24 = v21;
  if ( v21 )
    sub_B2FEA0((__int64)v21, v81, 0, 7, v77, (__int64)&v89, (v18 >> 2) & 7, v20, 0);
  v84 = v14;
  v87 = 0x400000000LL;
  v86 = v88;
  v78 = (__int64)v24;
  v89 = (const char *)v91;
  v91[0] = a1;
  v25 = (const char *)v91;
  v90 = 0x400000001LL;
  v26 = 1;
  do
  {
    v27 = v26--;
    v28 = *(_QWORD *)&v25[8 * v27 - 8];
    LODWORD(v90) = v26;
    v29 = *(_QWORD *)(v28 + 16);
    if ( v29 )
    {
      do
      {
        while ( 1 )
        {
          v32 = *(_BYTE **)(v29 + 24);
          if ( *v32 == 5 )
            break;
          v33 = (unsigned int)v87;
          v34 = (unsigned int)v87 + 1LL;
          if ( v34 > HIDWORD(v87) )
          {
            sub_C8D5F0((__int64)&v86, v88, v34, 8u, v22, v23);
            v33 = (unsigned int)v87;
          }
          *(_QWORD *)&v86[8 * v33] = v32;
          LODWORD(v87) = v87 + 1;
          v29 = *(_QWORD *)(v29 + 8);
          if ( !v29 )
            goto LABEL_16;
        }
        v30 = (unsigned int)v90;
        v31 = (unsigned int)v90 + 1LL;
        if ( v31 > HIDWORD(v90) )
        {
          sub_C8D5F0((__int64)&v89, v91, v31, 8u, v22, v23);
          v30 = (unsigned int)v90;
        }
        *(_QWORD *)&v89[8 * v30] = v32;
        LODWORD(v90) = v90 + 1;
        v29 = *(_QWORD *)(v29 + 8);
      }
      while ( v29 );
LABEL_16:
      v26 = v90;
      v25 = v89;
    }
  }
  while ( v26 );
  v35 = v84;
  v36 = v78;
  if ( v25 != (const char *)v91 )
    _libc_free((unsigned __int64)v25);
  v37 = 0;
  v38 = (__int64)&v86[8 * (unsigned int)v87];
  v85 = (_BYTE **)v86;
  v72 = (_BYTE **)v38;
  if ( v86 == (_BYTE *)v38 )
    goto LABEL_74;
  do
  {
    v39 = *v85;
    if ( **v85 == 62 )
    {
      v60 = **((_BYTE **)v39 - 8) != 20;
      v61 = (__int64 *)sub_BD5C60(a1);
      v62 = sub_ACD760(v61, v60);
      v38 = unk_3F10A10;
      v83 = v39[72];
      v75 = (*((_WORD *)v39 + 1) >> 7) & 7;
      v63 = sub_BD2C40(80, unk_3F10A10);
      if ( v63 )
      {
        v38 = v62;
        sub_B4D260((__int64)v63, v62, v36, 0, 0, v75, v83, (__int64)(v39 + 24), 0);
      }
    }
    else
    {
      for ( i = *((_QWORD *)v39 + 2); i; i = *((_QWORD *)v39 + 2) )
      {
        while ( 1 )
        {
          v43 = *(_QWORD *)(i + 24);
          if ( *(_BYTE *)v43 != 82 )
            break;
          v74 = *(_QWORD *)(v36 + 24);
          v89 = sub_BD5D20(v36);
          v91[0] = ".val";
          v92 = 773;
          v90 = v44;
          v82 = v39[72];
          v45 = *((_WORD *)v39 + 1) >> 7;
          v46 = sub_BD2C40(80, unk_3F10A14);
          v47 = v45 & 7;
          v48 = (__int64)v46;
          if ( v46 )
          {
            v79 = v46;
            sub_B4D0A0((__int64)v46, v74, v36, (__int64)&v89, 0, 0, v47, v82, (__int64)(v39 + 24), 0);
            v48 = (__int64)v79;
          }
          switch ( *(_WORD *)(v43 + 2) & 0x3F )
          {
            case ' ':
            case '%':
              v56 = v71;
              LOWORD(v56) = 0;
              v89 = "notinit";
              v92 = 259;
              v48 = sub_B50640(v48, (__int64)&v89, v43 + 24, v56);
              break;
            case '!':
            case '"':
              break;
            case '#':
              v59 = (__int64 *)sub_BD5C60(a1);
              v48 = sub_ACD6D0(v59);
              break;
            case '$':
              v58 = (__int64 *)sub_BD5C60(a1);
              v48 = sub_ACD720(v58);
              break;
            default:
              BUG();
          }
          v38 = v48;
          v37 = 1;
          sub_BD84D0(v43, v48);
          sub_B43D60((_QWORD *)v43);
          i = *((_QWORD *)v39 + 2);
          if ( !i )
            goto LABEL_53;
        }
        if ( *(_QWORD *)i )
        {
          v41 = *(_QWORD *)(i + 8);
          **(_QWORD **)(i + 16) = v41;
          if ( v41 )
            *(_QWORD *)(v41 + 16) = *(_QWORD *)(i + 16);
        }
        *(_QWORD *)i = v35;
        if ( v35 )
        {
          v42 = v35[2];
          *(_QWORD *)(i + 8) = v42;
          if ( v42 )
          {
            v38 = i + 8;
            *(_QWORD *)(v42 + 16) = i + 8;
          }
          *(_QWORD *)(i + 16) = v35 + 2;
          v35[2] = i;
        }
      }
    }
LABEL_53:
    sub_B43D60(v39);
    ++v85;
  }
  while ( v72 != v85 );
  if ( !v37 )
  {
LABEL_74:
    while ( 1 )
    {
      v49 = *(_QWORD *)(v36 + 16);
      if ( !v49 )
        break;
      sub_B43D60(*(_QWORD **)(v49 + 24));
    }
    sub_B30220(v36);
    *(_DWORD *)(v36 + 4) = *(_DWORD *)(v36 + 4) & 0xF8000000 | 1;
    sub_B2F9E0(v36, v38, v50, v51);
    sub_BD2DD0(v36);
  }
  else
  {
    sub_BA85C0(*(_QWORD *)(a1 + 40) + 8LL, v36);
    v57 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(v36 + 64) = a1 + 56;
    v57 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v36 + 56) = v57 | *(_QWORD *)(v36 + 56) & 7LL;
    *(_QWORD *)(v57 + 8) = v36 + 56;
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 56) & 7LL | (v36 + 56);
  }
  sub_B30290(a1);
  sub_B43D60(a2);
  v52 = v35[2];
  if ( !v52 )
    goto LABEL_62;
  while ( 1 )
  {
    v53 = *(unsigned __int8 **)(v52 + 24);
    v54 = *(_QWORD *)(v52 + 8);
    if ( *v53 > 0x1Cu )
    {
      v55 = sub_97D880(*(_QWORD *)(v52 + 24), a5, a6);
      if ( v55 )
        break;
    }
    if ( !v54 )
      goto LABEL_62;
LABEL_42:
    v52 = v54;
  }
  sub_BD84D0((__int64)v53, v55);
  if ( v54 )
  {
    while ( v53 == *(unsigned __int8 **)(v54 + 24) )
    {
      v54 = *(_QWORD *)(v54 + 8);
      if ( !v54 )
        goto LABEL_60;
    }
    if ( sub_F50EE0(v53, a6) )
      sub_B43D60(v53);
    goto LABEL_42;
  }
LABEL_60:
  if ( sub_F50EE0(v53, a6) )
    sub_B43D60(v53);
LABEL_62:
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
}
