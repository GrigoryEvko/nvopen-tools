// Function: sub_25BBDB0
// Address: 0x25bbdb0
//
__int64 __fastcall sub_25BBDB0(__int64 **a1, _QWORD *a2, char a3)
{
  unsigned int v3; // r13d
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rax
  const char *v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  const char *v10; // r14
  __int64 v11; // rax
  int v12; // ecx
  unsigned __int64 *v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // r15
  __int64 v16; // r14
  _BYTE *v17; // rax
  _BYTE *v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 (__fastcall *v23)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v24; // rax
  __int64 *v25; // r11
  _BYTE **v26; // rcx
  __int64 v27; // r14
  __int64 v28; // rax
  char v29; // al
  __int16 v30; // cx
  _QWORD *v31; // rax
  __int64 v32; // rbx
  unsigned __int64 *v33; // r14
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r10
  unsigned int v37; // ecx
  __int64 v38; // rdx
  __int64 v39; // rbx
  unsigned __int64 *v40; // r15
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rdx
  int v44; // eax
  char v45; // si
  int v46; // eax
  __int64 v47; // rax
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // r15
  __int64 v50; // [rsp+8h] [rbp-198h]
  unsigned int v51; // [rsp+14h] [rbp-18Ch]
  __int64 v53; // [rsp+20h] [rbp-180h]
  unsigned int v54; // [rsp+30h] [rbp-170h]
  __int16 v56; // [rsp+36h] [rbp-16Ah]
  __int64 v57; // [rsp+48h] [rbp-158h]
  _BYTE *v58; // [rsp+50h] [rbp-150h]
  __int64 v59; // [rsp+68h] [rbp-138h]
  _BYTE *v60; // [rsp+70h] [rbp-130h] BYREF
  __int64 v61; // [rsp+78h] [rbp-128h] BYREF
  _QWORD v62[4]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v63; // [rsp+A0h] [rbp-100h]
  const char *v64; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned int v65; // [rsp+B8h] [rbp-E8h]
  unsigned __int64 v66; // [rsp+C0h] [rbp-E0h]
  unsigned int v67; // [rsp+C8h] [rbp-D8h]
  __int16 v68; // [rsp+D0h] [rbp-D0h]
  unsigned __int64 *v69; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+E8h] [rbp-B8h]
  _BYTE v71[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+110h] [rbp-90h]
  __int64 v73; // [rsp+118h] [rbp-88h]
  __int64 v74; // [rsp+120h] [rbp-80h]
  _QWORD *v75; // [rsp+128h] [rbp-78h]
  void **v76; // [rsp+130h] [rbp-70h]
  void **v77; // [rsp+138h] [rbp-68h]
  __int64 v78; // [rsp+140h] [rbp-60h]
  int v79; // [rsp+148h] [rbp-58h]
  __int16 v80; // [rsp+14Ch] [rbp-54h]
  char v81; // [rsp+14Eh] [rbp-52h]
  __int64 v82; // [rsp+150h] [rbp-50h]
  __int64 v83; // [rsp+158h] [rbp-48h]
  void *v84; // [rsp+160h] [rbp-40h] BYREF
  void *v85; // [rsp+168h] [rbp-38h] BYREF

  v3 = 0;
  v59 = a2[2];
  if ( !v59 )
    return v3;
  do
  {
    v4 = *(_QWORD *)(v59 + 24);
    v59 = *(_QWORD *)(v59 + 8);
    if ( *(_BYTE *)v4 <= 0x1Cu )
      continue;
    v5 = (_QWORD *)sub_BD5C60(v4);
    v81 = 7;
    v75 = v5;
    v76 = &v84;
    v77 = &v85;
    v72 = 0;
    v78 = 0;
    v84 = &unk_49DA100;
    v73 = 0;
    v79 = 0;
    v80 = 512;
    v82 = 0;
    v83 = 0;
    LOWORD(v74) = 0;
    v85 = &unk_49DA0B0;
    v6 = *(_QWORD *)(v4 + 40);
    v69 = (unsigned __int64 *)v71;
    v72 = v6;
    v70 = 0x200000000LL;
    v73 = v4 + 24;
    v7 = *(const char **)sub_B46C60(v4);
    v64 = v7;
    if ( v7 && (sub_B96E90((__int64)&v64, (__int64)v7, 1), (v10 = v64) != 0) )
    {
      v11 = (__int64)v69;
      v12 = v70;
      v13 = &v69[2 * (unsigned int)v70];
      if ( v69 != v13 )
      {
        while ( *(_DWORD *)v11 )
        {
          v11 += 16;
          if ( v13 == (unsigned __int64 *)v11 )
            goto LABEL_38;
        }
        *(_QWORD *)(v11 + 8) = v64;
LABEL_10:
        sub_B91220((__int64)&v64, (__int64)v10);
        goto LABEL_11;
      }
LABEL_38:
      if ( (unsigned int)v70 >= (unsigned __int64)HIDWORD(v70) )
      {
        v48 = (unsigned int)v70 + 1LL;
        v49 = v50 & 0xFFFFFFFF00000000LL;
        v50 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v70) < v48 )
        {
          sub_C8D5F0((__int64)&v69, v71, v48, 0x10u, v8, v9);
          v13 = &v69[2 * (unsigned int)v70];
        }
        *v13 = v49;
        v13[1] = (unsigned __int64)v10;
        v10 = v64;
        LODWORD(v70) = v70 + 1;
      }
      else
      {
        if ( v13 )
        {
          *(_DWORD *)v13 = 0;
          v13[1] = (unsigned __int64)v10;
          v12 = v70;
          v10 = v64;
        }
        LODWORD(v70) = v12 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v69, 0);
      v10 = v64;
    }
    if ( v10 )
      goto LABEL_10;
LABEL_11:
    if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
      v14 = *(__int64 **)(v4 - 8);
    else
      v14 = (__int64 *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
    v15 = *v14;
    v16 = v14[4];
    v57 = **(_QWORD **)(*(_QWORD *)(v4 + 8) + 16LL);
    v58 = (_BYTE *)sub_ACD6D0(*a1);
    if ( !a3 )
    {
      v60 = (_BYTE *)v16;
      v63 = 257;
      v53 = sub_BCB2B0(v75);
      v23 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v76 + 8);
      if ( v23 == sub_920540 )
      {
        if ( sub_BCEA30(v53) )
          goto LABEL_42;
        if ( *(_BYTE *)v15 > 0x15u )
          goto LABEL_42;
        v24 = sub_25BBCF0(&v60, (__int64)&v61);
        if ( v26 != v24 )
          goto LABEL_42;
        LOBYTE(v68) = 0;
        v27 = sub_AD9FD0(v53, (unsigned __int8 *)v15, v25, 1, 0, (__int64)&v64, 0);
        if ( (_BYTE)v68 )
        {
          LOBYTE(v68) = 0;
          if ( v67 > 0x40 && v66 )
            j_j___libc_free_0_0(v66);
          if ( v65 > 0x40 && v64 )
            j_j___libc_free_0_0((unsigned __int64)v64);
        }
      }
      else
      {
        v27 = v23((__int64)v76, v53, (_BYTE *)v15, &v60, 1, 0);
      }
      if ( v27 )
      {
LABEL_28:
        v62[0] = "vfunc";
        v63 = 259;
        v28 = sub_AA4E30(v72);
        v29 = sub_AE5020(v28, v57);
        HIBYTE(v30) = HIBYTE(v56);
        v68 = 257;
        LOBYTE(v30) = v29;
        v56 = v30;
        v31 = sub_BD2C40(80, unk_3F10A14);
        v18 = v31;
        if ( v31 )
          sub_B4D190((__int64)v31, v57, v27, (__int64)&v64, 0, v56, 0, 0);
        (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v77 + 2))(v77, v18, v62, v73, v74);
        if ( v69 != &v69[2 * (unsigned int)v70] )
        {
          v32 = (__int64)v69;
          v33 = &v69[2 * (unsigned int)v70];
          do
          {
            v34 = *(_QWORD *)(v32 + 8);
            v35 = *(_DWORD *)v32;
            v32 += 16;
            sub_B99FD0((__int64)v18, v35, v34);
          }
          while ( v33 != (unsigned __int64 *)v32 );
        }
        goto LABEL_15;
      }
LABEL_42:
      v68 = 257;
      v27 = (__int64)sub_BD2C40(88, 2u);
      if ( !v27 )
        goto LABEL_45;
      v36 = *(_QWORD *)(v15 + 8);
      v37 = v54 & 0xE0000000 | 2;
      v54 = v37;
      if ( (unsigned int)*(unsigned __int8 *)(v36 + 8) - 17 <= 1 )
      {
LABEL_44:
        sub_B44260(v27, v36, 34, v37, 0, 0);
        *(_QWORD *)(v27 + 72) = v53;
        *(_QWORD *)(v27 + 80) = sub_B4DC50(v53, (__int64)&v60, 1);
        sub_B4D9A0(v27, v15, (__int64 *)&v60, 1, (__int64)&v64);
LABEL_45:
        sub_B4DDE0(v27, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v77 + 2))(v77, v27, v62, v73, v74);
        v38 = 2LL * (unsigned int)v70;
        if ( v69 != &v69[v38] )
        {
          v39 = (__int64)v69;
          v40 = &v69[v38];
          do
          {
            v41 = *(_QWORD *)(v39 + 8);
            v42 = *(_DWORD *)v39;
            v39 += 16;
            sub_B99FD0(v27, v42, v41);
          }
          while ( v40 != (unsigned __int64 *)v39 );
        }
        goto LABEL_28;
      }
      v43 = *((_QWORD *)v60 + 1);
      v44 = *(unsigned __int8 *)(v43 + 8);
      if ( v44 == 17 )
      {
        v45 = 0;
      }
      else
      {
        v45 = 1;
        if ( v44 != 18 )
          goto LABEL_44;
      }
      v46 = *(_DWORD *)(v43 + 32);
      BYTE4(v61) = v45;
      v51 = v37;
      LODWORD(v61) = v46;
      v47 = sub_BCE1B0((__int64 *)v36, v61);
      v37 = v51;
      v36 = v47;
      goto LABEL_44;
    }
    v62[0] = v15;
    v62[1] = v16;
    v64 = "rel_load";
    v68 = 259;
    v17 = *(_BYTE **)(v16 + 8);
    BYTE4(v61) = 0;
    v60 = v17;
    v18 = (_BYTE *)sub_B33D10((__int64)&v69, 0xD6u, (__int64)&v60, 1, (int)v62, 2, v61, (__int64)&v64);
LABEL_15:
    v19 = sub_ACADE0(*(__int64 ***)(v4 + 8));
    v68 = 257;
    LODWORD(v62[0]) = 1;
    v20 = sub_2466140((__int64 *)&v69, v19, v58, v62, 1, (__int64)&v64);
    v68 = 257;
    LODWORD(v62[0]) = 0;
    v21 = sub_2466140((__int64 *)&v69, v20, v18, v62, 1, (__int64)&v64);
    sub_BD84D0(v4, v21);
    sub_B43D60((_QWORD *)v4);
    nullsub_61();
    v84 = &unk_49DA100;
    nullsub_63();
    if ( v69 != (unsigned __int64 *)v71 )
      _libc_free((unsigned __int64)v69);
    v3 = 1;
  }
  while ( v59 );
  if ( (_BYTE)v3 )
    sub_B2E860(a2);
  return v3;
}
