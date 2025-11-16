// Function: sub_F369B0
// Address: 0xf369b0
//
__int64 __fastcall sub_F369B0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rax
  char *v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rbx
  int v18; // edx
  unsigned int v19; // ecx
  unsigned __int8 v20; // al
  int v21; // r14d
  char *v22; // r14
  char *v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rdx
  unsigned __int8 *v27; // rbx
  __int64 (__fastcall *v28)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v29; // r14
  const char *v30; // rax
  __int64 v31; // rdx
  __int64 (__fastcall *v32)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v33; // rbx
  _QWORD *v34; // rax
  __int64 v35; // r13
  __int64 v36; // rbx
  char *v37; // rbx
  char *v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // r13
  __int64 v45; // rbx
  char *v46; // rbx
  char *v47; // r12
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 *v50; // rax
  _QWORD **v51; // rdx
  int v52; // ecx
  __int64 *v53; // rax
  __int64 v54; // rsi
  char *v55; // r12
  char *v56; // r13
  __int64 v57; // rdx
  unsigned int v58; // esi
  __int64 v59; // [rsp+0h] [rbp-190h]
  int v60; // [rsp+8h] [rbp-188h]
  __int64 v61; // [rsp+30h] [rbp-160h]
  __int64 v62; // [rsp+38h] [rbp-158h]
  __int64 v64; // [rsp+40h] [rbp-150h]
  __int64 v65; // [rsp+50h] [rbp-140h]
  __int64 v66; // [rsp+58h] [rbp-138h]
  __int64 v67; // [rsp+68h] [rbp-128h]
  const char *v68; // [rsp+70h] [rbp-120h] BYREF
  __int64 v69; // [rsp+78h] [rbp-118h]
  char *v70; // [rsp+80h] [rbp-110h]
  __int16 v71; // [rsp+90h] [rbp-100h]
  const char *v72[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v73; // [rsp+C0h] [rbp-D0h]
  char *v74; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-B8h]
  _BYTE v76[16]; // [rsp+E0h] [rbp-B0h] BYREF
  __int16 v77; // [rsp+F0h] [rbp-A0h]
  __int64 v78; // [rsp+100h] [rbp-90h]
  __int64 v79; // [rsp+108h] [rbp-88h]
  __int64 v80; // [rsp+110h] [rbp-80h]
  __int64 v81; // [rsp+118h] [rbp-78h]
  void **v82; // [rsp+120h] [rbp-70h]
  void **v83; // [rsp+128h] [rbp-68h]
  __int64 v84; // [rsp+130h] [rbp-60h]
  int v85; // [rsp+138h] [rbp-58h]
  __int16 v86; // [rsp+13Ch] [rbp-54h]
  char v87; // [rsp+13Eh] [rbp-52h]
  __int64 v88; // [rsp+140h] [rbp-50h]
  __int64 v89; // [rsp+148h] [rbp-48h]
  void *v90; // [rsp+150h] [rbp-40h] BYREF
  void *v91; // [rsp+158h] [rbp-38h] BYREF

  if ( !a2 )
    BUG();
  v4 = a2[2];
  v77 = 257;
  v61 = v4;
  v6 = sub_F36990(v4, a2, a3, 0, 0, 0, (void **)&v74, 0);
  v7 = a2[2];
  v77 = 257;
  v8 = v6;
  v62 = sub_F36990(v7, a2, a3, 0, 0, 0, (void **)&v74, 0);
  v66 = *(_QWORD *)(a1 + 8);
  v9 = sub_B43CC0((__int64)(a2 - 3));
  v10 = (char *)sub_9208B0(v9, v66);
  v75 = v11;
  v74 = v10;
  v60 = sub_CA1930(&v74);
  v12 = sub_986580(v8);
  v13 = sub_BD5C60(v12);
  v86 = 512;
  v81 = v13;
  v82 = &v90;
  v83 = &v91;
  v74 = v76;
  v90 = &unk_49DA100;
  v75 = 0x200000000LL;
  v84 = 0;
  v91 = &unk_49DA0B0;
  v85 = 0;
  v87 = 7;
  v88 = 0;
  v89 = 0;
  v78 = 0;
  v79 = 0;
  LOWORD(v80) = 0;
  sub_D5F1F0((__int64)&v74, v12);
  v68 = "iv";
  v71 = 259;
  v73 = 257;
  v14 = sub_BD2DA0(80);
  v15 = v14;
  if ( v14 )
  {
    v16 = v14;
    sub_B44260(v14, v66, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v15 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v15, v72);
    sub_BD2A10(v15, *(_DWORD *)(v15 + 72), 1);
  }
  else
  {
    v16 = 0;
  }
  if ( *(_BYTE *)v15 > 0x1Cu )
  {
    switch ( *(_BYTE *)v15 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_9;
      case 'T':
      case 'U':
      case 'V':
        v17 = *(_QWORD *)(v15 + 8);
        v18 = *(unsigned __int8 *)(v17 + 8);
        v19 = v18 - 17;
        v20 = *(_BYTE *)(v17 + 8);
        if ( (unsigned int)(v18 - 17) <= 1 )
          v20 = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
        if ( v20 <= 3u || v20 == 5 || (v20 & 0xFD) == 4 )
          goto LABEL_9;
        if ( (_BYTE)v18 == 15 )
        {
          if ( (*(_BYTE *)(v17 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v15 + 8)) )
            break;
          v50 = *(__int64 **)(v17 + 16);
          v17 = *v50;
          v18 = *(unsigned __int8 *)(*v50 + 8);
          v19 = v18 - 17;
        }
        else if ( (_BYTE)v18 == 16 )
        {
          do
          {
            v17 = *(_QWORD *)(v17 + 24);
            LOBYTE(v18) = *(_BYTE *)(v17 + 8);
          }
          while ( (_BYTE)v18 == 16 );
          v19 = (unsigned __int8)v18 - 17;
        }
        if ( v19 <= 1 )
          LOBYTE(v18) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
        if ( (unsigned __int8)v18 <= 3u || (_BYTE)v18 == 5 || (v18 & 0xFD) == 4 )
        {
LABEL_9:
          v21 = v85;
          if ( v84 )
            sub_B99FD0(v15, 3u, v84);
          sub_B45150(v15, v21);
        }
        break;
      default:
        break;
    }
  }
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v83 + 2))(v83, v15, &v68, v79, v80);
  v22 = v74;
  v23 = &v74[16 * (unsigned int)v75];
  if ( v74 != v23 )
  {
    do
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *(_DWORD *)v22;
      v22 += 16;
      sub_B99FD0(v15, v25, v24);
    }
    while ( v23 != v22 );
  }
  v68 = sub_BD5D20(v16);
  v69 = v26;
  v71 = 773;
  v70 = ".next";
  v27 = (unsigned __int8 *)sub_AD64C0(v66, 1, 0);
  v28 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v82 + 4);
  if ( v28 != sub_9201A0 )
  {
    v29 = v28((__int64)v82, 13u, (_BYTE *)v15, v27, 1u, v60 != 2);
    goto LABEL_19;
  }
  if ( *(_BYTE *)v15 <= 0x15u && *v27 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
      v29 = sub_AD5570(13, v15, v27, 2 * (v60 != 2) + 1, 0);
    else
      v29 = sub_AABE40(0xDu, (unsigned __int8 *)v15, v27);
LABEL_19:
    if ( v29 )
      goto LABEL_20;
  }
  v73 = 257;
  v29 = sub_B504D0(13, v15, (__int64)v27, (__int64)v72, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v83 + 2))(v83, v29, &v68, v79, v80);
  v45 = 16LL * (unsigned int)v75;
  if ( v74 != &v74[v45] )
  {
    v59 = v15;
    v46 = &v74[v45];
    v47 = v74;
    do
    {
      v48 = *((_QWORD *)v47 + 1);
      v49 = *(_DWORD *)v47;
      v47 += 16;
      sub_B99FD0(v29, v49, v48);
    }
    while ( v46 != v47 );
    v15 = v59;
  }
  sub_B447F0((unsigned __int8 *)v29, 1);
  if ( v60 != 2 )
    sub_B44850((unsigned __int8 *)v29, 1);
LABEL_20:
  v30 = sub_BD5D20(v16);
  v71 = 773;
  v68 = v30;
  v69 = v31;
  v70 = ".check";
  v32 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, _BYTE *))*((_QWORD *)*v82 + 7);
  if ( v32 != sub_928890 )
  {
    v33 = v32((__int64)v82, 32u, (_BYTE *)v29, (_BYTE *)a1);
LABEL_24:
    if ( v33 )
      goto LABEL_25;
    goto LABEL_54;
  }
  if ( *(_BYTE *)v29 <= 0x15u && *(_BYTE *)a1 <= 0x15u )
  {
    v33 = sub_AAB310(0x20u, (unsigned __int8 *)v29, (unsigned __int8 *)a1);
    goto LABEL_24;
  }
LABEL_54:
  v73 = 257;
  v33 = (__int64)sub_BD2C40(72, unk_3F10FD0);
  if ( v33 )
  {
    v51 = *(_QWORD ***)(v29 + 8);
    v52 = *((unsigned __int8 *)v51 + 8);
    if ( (unsigned int)(v52 - 17) > 1 )
    {
      v54 = sub_BCB2A0(*v51);
    }
    else
    {
      BYTE4(v67) = (_BYTE)v52 == 18;
      LODWORD(v67) = *((_DWORD *)v51 + 8);
      v53 = (__int64 *)sub_BCB2A0(*v51);
      v54 = sub_BCE1B0(v53, v67);
    }
    sub_B523C0(v33, v54, 53, 32, v29, a1, (__int64)v72, 0, 0, 0);
  }
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v83 + 2))(v83, v33, &v68, v79, v80);
  if ( v74 != &v74[16 * (unsigned int)v75] )
  {
    v64 = v15;
    v55 = v74;
    v56 = &v74[16 * (unsigned int)v75];
    do
    {
      v57 = *((_QWORD *)v55 + 1);
      v58 = *(_DWORD *)v55;
      v55 += 16;
      sub_B99FD0(v33, v58, v57);
    }
    while ( v56 != v55 );
    v15 = v64;
  }
LABEL_25:
  v73 = 257;
  v34 = sub_BD2C40(72, 3u);
  v35 = (__int64)v34;
  if ( v34 )
    sub_B4C9A0((__int64)v34, v62, v8, v33, 3u, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v83 + 2))(v83, v35, v72, v79, v80);
  v36 = 16LL * (unsigned int)v75;
  if ( v74 != &v74[v36] )
  {
    v65 = v15;
    v37 = &v74[v36];
    v38 = v74;
    do
    {
      v39 = *((_QWORD *)v38 + 1);
      v40 = *(_DWORD *)v38;
      v38 += 16;
      sub_B99FD0(v35, v40, v39);
    }
    while ( v37 != v38 );
    v15 = v65;
  }
  v41 = (_QWORD *)sub_986580(v8);
  sub_B43D60(v41);
  v42 = sub_AD64C0(v66, 0, 0);
  sub_F0A850(v15, v42, v61);
  sub_F0A850(v15, v29, v8);
  v43 = sub_AA4FF0(v8);
  if ( v43 )
    v43 -= 24;
  nullsub_61();
  v90 = &unk_49DA100;
  nullsub_63();
  if ( v74 != v76 )
    _libc_free(v74, v29);
  return v43;
}
