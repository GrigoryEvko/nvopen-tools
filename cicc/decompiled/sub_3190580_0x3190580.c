// Function: sub_3190580
// Address: 0x3190580
//
__int64 *__fastcall sub_3190580(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 v14; // rax
  int v15; // ecx
  _QWORD *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // rax
  int v22; // ecx
  _QWORD *v23; // rdx
  __int64 v24; // rax
  unsigned __int64 *v25; // rax
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rdx
  __int64 **v28; // rcx
  unsigned __int64 v29; // rax
  __int64 **v30; // rcx
  unsigned __int8 *v31; // r15
  unsigned __int8 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r11
  __int64 (__fastcall *v35)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v36; // rax
  __int64 v37; // r14
  unsigned __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r13
  _BYTE *v42; // r12
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 v46; // r15
  _BYTE *v47; // r15
  __int64 v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // rax
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rsi
  unsigned __int8 *v54; // [rsp+8h] [rbp-168h]
  __int64 v55; // [rsp+8h] [rbp-168h]
  unsigned __int64 v56; // [rsp+20h] [rbp-150h]
  int v58[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v59; // [rsp+70h] [rbp-100h]
  _QWORD v60[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v61; // [rsp+A0h] [rbp-D0h]
  _BYTE *v62; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+B8h] [rbp-B8h]
  _BYTE v64[16]; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v65; // [rsp+D0h] [rbp-A0h]
  __int64 v66; // [rsp+E0h] [rbp-90h]
  __int64 v67; // [rsp+E8h] [rbp-88h]
  __int64 v68; // [rsp+F0h] [rbp-80h]
  __int64 v69; // [rsp+F8h] [rbp-78h]
  void **v70; // [rsp+100h] [rbp-70h]
  void **v71; // [rsp+108h] [rbp-68h]
  __int64 v72; // [rsp+110h] [rbp-60h]
  int v73; // [rsp+118h] [rbp-58h]
  __int16 v74; // [rsp+11Ch] [rbp-54h]
  char v75; // [rsp+11Eh] [rbp-52h]
  __int64 v76; // [rsp+120h] [rbp-50h]
  __int64 v77; // [rsp+128h] [rbp-48h]
  void *v78; // [rsp+130h] [rbp-40h] BYREF
  void *v79; // [rsp+138h] [rbp-38h] BYREF

  v3 = a2;
  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 72LL);
  v65 = 257;
  v5 = sub_B2BE50(v4);
  v6 = sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
    sub_AA4D50(v6, v5, (__int64)&v62, v4, a3);
  v8 = *(_QWORD *)(v7 + 56);
  *a1 = v7;
  v9 = sub_AA48A0(v7);
  v75 = 7;
  v69 = v9;
  v70 = &v78;
  v71 = &v79;
  v74 = 512;
  v62 = v64;
  v78 = &unk_49DA100;
  v66 = v7;
  v63 = 0x200000000LL;
  v79 = &unk_49DA0B0;
  LOWORD(v68) = 1;
  v72 = 0;
  v73 = 0;
  v76 = 0;
  v77 = 0;
  v67 = v8;
  if ( v8 != v7 + 48 )
  {
    if ( v8 )
      v8 -= 24;
    v10 = *(_QWORD *)sub_B46C60(v8);
    v60[0] = v10;
    if ( v10 && (sub_B96E90((__int64)v60, v10, 1), (v13 = v60[0]) != 0) )
    {
      v14 = (__int64)v62;
      v15 = v63;
      v16 = &v62[16 * (unsigned int)v63];
      if ( v62 != (_BYTE *)v16 )
      {
        while ( *(_DWORD *)v14 )
        {
          v14 += 16;
          if ( v16 == (_QWORD *)v14 )
            goto LABEL_53;
        }
        *(_QWORD *)(v14 + 8) = v60[0];
LABEL_13:
        sub_B91220((__int64)v60, v13);
        goto LABEL_14;
      }
LABEL_53:
      if ( (unsigned int)v63 >= (unsigned __int64)HIDWORD(v63) )
      {
        v53 = (unsigned int)v63 + 1LL;
        if ( HIDWORD(v63) < v53 )
        {
          sub_C8D5F0((__int64)&v62, v64, v53, 0x10u, v11, v12);
          v16 = &v62[16 * (unsigned int)v63];
        }
        *v16 = 0;
        v16[1] = v13;
        v13 = v60[0];
        LODWORD(v63) = v63 + 1;
      }
      else
      {
        if ( v16 )
        {
          *(_DWORD *)v16 = 0;
          v16[1] = v13;
          v15 = v63;
          v13 = v60[0];
        }
        LODWORD(v63) = v15 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v62, 0);
      v13 = v60[0];
    }
    if ( !v13 )
      goto LABEL_14;
    goto LABEL_13;
  }
LABEL_14:
  v17 = *(_QWORD *)(*(_QWORD *)(v3 + 8) + 48LL);
  v60[0] = v17;
  if ( v17 && (sub_B96E90((__int64)v60, v17, 1), (v20 = v60[0]) != 0) )
  {
    v21 = (__int64)v62;
    v22 = v63;
    v23 = &v62[16 * (unsigned int)v63];
    if ( v62 != (_BYTE *)v23 )
    {
      while ( *(_DWORD *)v21 )
      {
        v21 += 16;
        if ( v23 == (_QWORD *)v21 )
          goto LABEL_45;
      }
      *(_QWORD *)(v21 + 8) = v60[0];
      goto LABEL_21;
    }
LABEL_45:
    if ( (unsigned int)v63 >= (unsigned __int64)HIDWORD(v63) )
    {
      v52 = (unsigned int)v63 + 1LL;
      if ( HIDWORD(v63) < v52 )
      {
        sub_C8D5F0((__int64)&v62, v64, v52, 0x10u, v18, v19);
        v23 = &v62[16 * (unsigned int)v63];
      }
      *v23 = 0;
      v23[1] = v20;
      v20 = v60[0];
      LODWORD(v63) = v63 + 1;
    }
    else
    {
      if ( v23 )
      {
        *(_DWORD *)v23 = 0;
        v23[1] = v20;
        v22 = v63;
        v20 = v60[0];
      }
      LODWORD(v63) = v22 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v62, 0);
    v20 = v60[0];
  }
  if ( v20 )
LABEL_21:
    sub_B91220((__int64)v60, v20);
  v24 = *(_QWORD *)(v3 + 8);
  if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
    v25 = *(unsigned __int64 **)(v24 - 8);
  else
    v25 = (unsigned __int64 *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF));
  v26 = *v25;
  v27 = v25[4];
  v28 = *(__int64 ***)(v3 + 16);
  v61 = 257;
  v29 = sub_318FC60((__int64 *)&v62, 0x26u, v27, v28, (__int64)v60, 0, v58[0], 0);
  v30 = *(__int64 ***)(v3 + 16);
  v31 = (unsigned __int8 *)v29;
  v61 = 257;
  v32 = (unsigned __int8 *)sub_318FC60((__int64 *)&v62, 0x26u, v26, v30, (__int64)v60, 0, v58[0], 0);
  v61 = 257;
  v54 = v32;
  v33 = sub_3122580((__int64 *)&v62, v32, v31, (__int64)v60, 0);
  v34 = (__int64)v54;
  v59 = 257;
  v56 = v33;
  v35 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v70 + 2);
  if ( v35 != sub_9202E0 )
  {
    v51 = v35((__int64)v70, 22u, v54, v31);
    v34 = (__int64)v54;
    v37 = v51;
    goto LABEL_30;
  }
  if ( *v54 <= 0x15u && *v31 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(22) )
      v36 = sub_AD5570(22, (__int64)v54, v31, 0, 0);
    else
      v36 = sub_AABE40(0x16u, v54, v31);
    v34 = (__int64)v54;
    v37 = v36;
LABEL_30:
    if ( v37 )
      goto LABEL_31;
  }
  v61 = 257;
  v37 = sub_B504D0(22, v34, (__int64)v31, (__int64)v60, 0, 0);
  (*((void (__fastcall **)(void **, __int64, int *, __int64, __int64))*v71 + 2))(v71, v37, v58, v67, v68);
  v46 = 16LL * (unsigned int)v63;
  if ( v62 != &v62[v46] )
  {
    v55 = v3;
    v47 = &v62[v46];
    v48 = (__int64)v62;
    do
    {
      v49 = *(_QWORD *)(v48 + 8);
      v50 = *(_DWORD *)v48;
      v48 += 16;
      sub_B99FD0(v37, v50, v49);
    }
    while ( v47 != (_BYTE *)v48 );
    v3 = v55;
  }
LABEL_31:
  v61 = 257;
  v38 = sub_318FC60((__int64 *)&v62, 0x27u, v56, *(__int64 ***)(*(_QWORD *)(v3 + 8) + 8LL), (__int64)v60, 0, v58[0], 0);
  v61 = 257;
  a1[1] = v38;
  a1[2] = sub_318FC60(
            (__int64 *)&v62,
            0x27u,
            v37,
            *(__int64 ***)(*(_QWORD *)(v3 + 8) + 8LL),
            (__int64)v60,
            0,
            v58[0],
            0);
  v61 = 257;
  v39 = sub_BD2C40(72, 1u);
  v40 = (__int64)v39;
  if ( v39 )
    sub_B4C8F0((__int64)v39, a3, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v71 + 2))(v71, v40, v60, v67, v68);
  v41 = (__int64)v62;
  v42 = &v62[16 * (unsigned int)v63];
  if ( v62 != v42 )
  {
    do
    {
      v43 = *(_QWORD *)(v41 + 8);
      v44 = *(_DWORD *)v41;
      v41 += 16;
      sub_B99FD0(v40, v44, v43);
    }
    while ( v42 != (_BYTE *)v41 );
  }
  nullsub_61();
  v78 = &unk_49DA100;
  nullsub_63();
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  return a1;
}
