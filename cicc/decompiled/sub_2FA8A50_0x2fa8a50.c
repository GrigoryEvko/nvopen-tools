// Function: sub_2FA8A50
// Address: 0x2fa8a50
//
void __fastcall sub_2FA8A50(__int64 *a1, __int64 a2, int a3)
{
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r13
  unsigned int *v11; // rax
  int v12; // ecx
  unsigned int *v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r15
  _BYTE *v16; // r13
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v21; // rax
  __int64 *v22; // r10
  _BYTE **v23; // rcx
  __int64 v24; // r13
  __int64 v25; // r15
  __int64 v26; // rax
  char v27; // al
  _QWORD *v28; // rax
  __int64 v29; // r9
  __int64 v30; // rbx
  unsigned __int64 v31; // r15
  unsigned int *v32; // r12
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // r11
  unsigned __int64 v36; // rbx
  unsigned int *v37; // r15
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // rax
  int v41; // edx
  int v42; // edx
  char v43; // dl
  unsigned __int64 v44; // rsi
  char v46; // [rsp+20h] [rbp-150h]
  __int64 v47; // [rsp+28h] [rbp-148h]
  __int64 v48; // [rsp+38h] [rbp-138h]
  _BYTE *v49; // [rsp+40h] [rbp-130h] BYREF
  __int64 v50; // [rsp+48h] [rbp-128h]
  _QWORD v51[4]; // [rsp+50h] [rbp-120h] BYREF
  char v52; // [rsp+70h] [rbp-100h]
  char v53; // [rsp+71h] [rbp-FFh]
  __int64 v54; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v55; // [rsp+88h] [rbp-E8h]
  unsigned __int64 v56; // [rsp+90h] [rbp-E0h]
  unsigned int v57; // [rsp+98h] [rbp-D8h]
  __int16 v58; // [rsp+A0h] [rbp-D0h]
  unsigned int *v59; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+B8h] [rbp-B8h]
  _BYTE v61[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+E0h] [rbp-90h]
  __int64 v63; // [rsp+E8h] [rbp-88h]
  __int64 v64; // [rsp+F0h] [rbp-80h]
  __int64 v65; // [rsp+F8h] [rbp-78h]
  void **v66; // [rsp+100h] [rbp-70h]
  void **v67; // [rsp+108h] [rbp-68h]
  __int64 v68; // [rsp+110h] [rbp-60h]
  int v69; // [rsp+118h] [rbp-58h]
  __int16 v70; // [rsp+11Ch] [rbp-54h]
  char v71; // [rsp+11Eh] [rbp-52h]
  __int64 v72; // [rsp+120h] [rbp-50h]
  __int64 v73; // [rsp+128h] [rbp-48h]
  void *v74; // [rsp+130h] [rbp-40h] BYREF
  void *v75; // [rsp+138h] [rbp-38h] BYREF

  v4 = a1;
  v5 = sub_BD5C60(a2);
  v67 = &v75;
  v65 = v5;
  v66 = &v74;
  v59 = (unsigned int *)v61;
  v74 = &unk_49DA100;
  v60 = 0x200000000LL;
  v68 = 0;
  v75 = &unk_49DA0B0;
  v6 = *(_QWORD *)(a2 + 40);
  v69 = 0;
  v62 = v6;
  v70 = 512;
  v71 = 7;
  v72 = 0;
  v73 = 0;
  v63 = a2 + 24;
  LOWORD(v64) = 0;
  v7 = *(_QWORD *)sub_B46C60(a2);
  v54 = v7;
  if ( v7 && (sub_B96E90((__int64)&v54, v7, 1), (v10 = v54) != 0) )
  {
    v11 = v59;
    v12 = v60;
    v13 = &v59[4 * (unsigned int)v60];
    if ( v59 != v13 )
    {
      while ( 1 )
      {
        v9 = *v11;
        if ( !(_DWORD)v9 )
          break;
        v11 += 4;
        if ( v13 == v11 )
          goto LABEL_25;
      }
      *((_QWORD *)v11 + 1) = v54;
      goto LABEL_8;
    }
LABEL_25:
    if ( (unsigned int)v60 >= (unsigned __int64)HIDWORD(v60) )
    {
      v44 = (unsigned int)v60 + 1LL;
      if ( HIDWORD(v60) < v44 )
      {
        sub_C8D5F0((__int64)&v59, v61, v44, 0x10u, v8, v9);
        v13 = &v59[4 * (unsigned int)v60];
      }
      *(_QWORD *)v13 = 0;
      *((_QWORD *)v13 + 1) = v10;
      v10 = v54;
      LODWORD(v60) = v60 + 1;
    }
    else
    {
      if ( v13 )
      {
        *v13 = 0;
        *((_QWORD *)v13 + 1) = v10;
        v12 = v60;
        v10 = v54;
      }
      LODWORD(v60) = v12 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v59, 0);
    v10 = v54;
  }
  if ( v10 )
LABEL_8:
    sub_B91220((__int64)&v54, v10);
  v14 = (_QWORD *)sub_BD5C60(a2);
  v15 = sub_BCB2D0(v14);
  v16 = (_BYTE *)sub_AD64C0(v15, 0, 0);
  v17 = sub_AD64C0(v15, 1, 0);
  v18 = a1[3];
  v49 = v16;
  v50 = v17;
  v51[0] = "call_site";
  v19 = a1[15];
  v53 = 1;
  v52 = 3;
  v47 = v19;
  v20 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v66 + 8);
  if ( v20 == sub_920540 )
  {
    if ( sub_BCEA30(v18) )
      goto LABEL_29;
    if ( *(_BYTE *)v47 > 0x15u )
      goto LABEL_29;
    v21 = sub_2FA8680(&v49, (__int64)v51);
    if ( v23 != v21 )
      goto LABEL_29;
    LOBYTE(v58) = 0;
    v24 = sub_AD9FD0(v18, (unsigned __int8 *)v47, v22, 2, 0, (__int64)&v54, 0);
    if ( (_BYTE)v58 )
    {
      LOBYTE(v58) = 0;
      if ( v57 > 0x40 && v56 )
        j_j___libc_free_0_0(v56);
      if ( v55 > 0x40 && v54 )
        j_j___libc_free_0_0(v54);
    }
  }
  else
  {
    v24 = v20((__int64)v66, v18, (_BYTE *)v47, &v49, 2, 0);
  }
  if ( v24 )
    goto LABEL_15;
LABEL_29:
  v58 = 257;
  v24 = (__int64)sub_BD2C40(88, 3u);
  if ( !v24 )
    goto LABEL_32;
  v35 = *(_QWORD *)(v47 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 > 1 )
  {
    v40 = *((_QWORD *)v49 + 1);
    v41 = *(unsigned __int8 *)(v40 + 8);
    if ( v41 != 17 )
    {
      if ( v41 == 18 )
      {
LABEL_41:
        v43 = 1;
LABEL_43:
        BYTE4(v48) = v43;
        LODWORD(v48) = *(_DWORD *)(v40 + 32);
        v35 = sub_BCE1B0((__int64 *)v35, v48);
        goto LABEL_31;
      }
      v40 = *(_QWORD *)(v50 + 8);
      v42 = *(unsigned __int8 *)(v40 + 8);
      if ( v42 != 17 )
      {
        if ( v42 != 18 )
          goto LABEL_31;
        goto LABEL_41;
      }
    }
    v43 = 0;
    goto LABEL_43;
  }
LABEL_31:
  sub_B44260(v24, v35, 34, 3u, 0, 0);
  *(_QWORD *)(v24 + 72) = v18;
  *(_QWORD *)(v24 + 80) = sub_B4DC50(v18, (__int64)&v49, 2);
  sub_B4D9A0(v24, v47, (__int64 *)&v49, 2, (__int64)&v54);
LABEL_32:
  sub_B4DDE0(v24, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v67 + 2))(v67, v24, v51, v63, v64);
  if ( v59 != &v59[4 * (unsigned int)v60] )
  {
    v36 = (unsigned __int64)v59;
    v37 = &v59[4 * (unsigned int)v60];
    do
    {
      v38 = *(_QWORD *)(v36 + 8);
      v39 = *(_DWORD *)v36;
      v36 += 16LL;
      sub_B99FD0(v24, v39, v38);
    }
    while ( v37 != (unsigned int *)v36 );
    v4 = a1;
  }
LABEL_15:
  v25 = sub_ACD640(*v4, a3, 0);
  v26 = sub_AA4E30(v62);
  v27 = sub_AE5020(v26, *(_QWORD *)(v25 + 8));
  v58 = 257;
  v46 = v27;
  v28 = sub_BD2C40(80, unk_3F10A10);
  v30 = (__int64)v28;
  if ( v28 )
    sub_B4D3C0((__int64)v28, v25, v24, 1, v46, v29, 0, 0);
  (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v67 + 2))(v67, v30, &v54, v63, v64);
  v31 = (unsigned __int64)v59;
  v32 = &v59[4 * (unsigned int)v60];
  if ( v59 != v32 )
  {
    do
    {
      v33 = *(_QWORD *)(v31 + 8);
      v34 = *(_DWORD *)v31;
      v31 += 16LL;
      sub_B99FD0(v30, v34, v33);
    }
    while ( v32 != (unsigned int *)v31 );
  }
  nullsub_61();
  v74 = &unk_49DA100;
  nullsub_63();
  if ( v59 != (unsigned int *)v61 )
    _libc_free((unsigned __int64)v59);
}
