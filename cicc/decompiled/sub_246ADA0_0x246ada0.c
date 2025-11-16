// Function: sub_246ADA0
// Address: 0x246ada0
//
void __fastcall sub_246ADA0(__int64 a1)
{
  unsigned int v1; // r15d
  __int64 v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  __int64 v6; // rdx
  _QWORD **v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int64 **v11; // rbx
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // r12
  _BYTE *v14; // rax
  __int64 v15; // r11
  __int64 v16; // r14
  __int64 (__fastcall *v17)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  char v21; // al
  __int16 v22; // cx
  _QWORD *v23; // rax
  __int64 v24; // r12
  unsigned int *v25; // r14
  unsigned int *v26; // rbx
  __int64 v27; // rdx
  unsigned int v28; // esi
  char v29; // bl
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned __int16 v35; // ax
  __int64 v36; // rdx
  unsigned int v37; // eax
  unsigned int v38; // eax
  int v39; // r12d
  unsigned int *v40; // r14
  unsigned int *v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  int v44; // r14d
  unsigned int *v45; // r14
  unsigned int *v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rsi
  unsigned __int16 v54; // bx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // r9
  unsigned int v60; // ecx
  unsigned int v61; // eax
  _QWORD **v62; // [rsp+30h] [rbp-230h]
  __int64 v63; // [rsp+38h] [rbp-228h]
  __int64 **v64; // [rsp+48h] [rbp-218h]
  __int64 v65; // [rsp+48h] [rbp-218h]
  __int64 v66; // [rsp+48h] [rbp-218h]
  unsigned __int16 v67; // [rsp+52h] [rbp-20Eh]
  __int16 v68; // [rsp+56h] [rbp-20Ah]
  _QWORD **v69; // [rsp+68h] [rbp-1F8h]
  _BYTE v70[32]; // [rsp+70h] [rbp-1F0h] BYREF
  __int16 v71; // [rsp+90h] [rbp-1D0h]
  _QWORD v72[4]; // [rsp+A0h] [rbp-1C0h] BYREF
  __int16 v73; // [rsp+C0h] [rbp-1A0h]
  _QWORD *v74; // [rsp+D0h] [rbp-190h] BYREF
  _QWORD v75[2]; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v76; // [rsp+F0h] [rbp-170h]
  __int64 v77; // [rsp+F8h] [rbp-168h]
  __int64 v78; // [rsp+100h] [rbp-160h]
  unsigned int *v79[9]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD *v80; // [rsp+158h] [rbp-108h]
  unsigned int *v81; // [rsp+1A0h] [rbp-C0h] BYREF
  int v82; // [rsp+1A8h] [rbp-B8h]
  char v83; // [rsp+1B0h] [rbp-B0h] BYREF
  __int16 v84; // [rsp+1C0h] [rbp-A0h]
  __int64 v85; // [rsp+1D0h] [rbp-90h]
  __int64 v86; // [rsp+1D8h] [rbp-88h]
  __int64 v87; // [rsp+1E0h] [rbp-80h]
  _QWORD *v88; // [rsp+1E8h] [rbp-78h]
  __int64 v89; // [rsp+1F0h] [rbp-70h]
  __int64 v90; // [rsp+1F8h] [rbp-68h]
  __int64 v91; // [rsp+200h] [rbp-60h]
  int v92; // [rsp+208h] [rbp-58h]
  void *v93; // [rsp+220h] [rbp-40h]

  sub_23D0AB0((__int64)v79, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL), 0, 0, 0);
  v84 = 257;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 152LL);
  v3 = sub_BCB2E0(v80);
  v63 = sub_A82CA0(v79, v3, v2, 0, 0, (__int64)&v81);
  *(_QWORD *)(a1 + 192) = v63;
  if ( *(_DWORD *)(a1 + 40) )
  {
    v50 = *(_QWORD *)(a1 + 16);
    v84 = 257;
    v51 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v50 + 72));
    v52 = sub_23DEB90((__int64 *)v79, v51, v63, (__int64)&v81);
    v53 = (unsigned __int8)byte_4FE8EA8;
    *(_QWORD *)(a1 + 184) = v52;
    LOBYTE(v54) = v53;
    HIBYTE(v54) = 1;
    *(_WORD *)(v52 + 2) = v53 | *(_WORD *)(v52 + 2) & 0xFFC0;
    v55 = sub_BCB2B0(v80);
    v56 = sub_AD6530(v55, v53);
    sub_B34240((__int64)v79, *(_QWORD *)(a1 + 184), v56, v63, v54, 0, 0, 0, 0);
    v84 = 257;
    HIDWORD(v74) = 0;
    v57 = sub_BCB2E0(v80);
    v58 = sub_ACD640(v57, 800, 0);
    v59 = sub_B33C40((__int64)v79, 0x16Eu, v63, v58, (unsigned int)v74, (__int64)&v81);
    v60 = (unsigned __int8)byte_4FE8EA8;
    v61 = (unsigned __int8)byte_4FE8EA8;
    BYTE1(v60) = 1;
    BYTE1(v61) = 1;
    sub_B343C0(
      (__int64)v79,
      0xEEu,
      *(_QWORD *)(a1 + 184),
      v60,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
      v61,
      v59,
      0,
      0,
      0,
      0,
      0);
  }
  v4 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL);
  v74 = v75;
  v5 = v4[29];
  sub_2462160((__int64 *)&v74, (_BYTE *)v5, v5 + v4[30]);
  v6 = *(unsigned int *)(a1 + 40);
  v76 = v4[33];
  v77 = v4[34];
  v78 = v4[35];
  v7 = *(_QWORD ***)(a1 + 32);
  v62 = &v7[v6];
  if ( v62 != v7 )
  {
    v69 = *(_QWORD ***)(a1 + 32);
    while ( 1 )
    {
      v8 = (__int64)*v69;
      sub_2468350((__int64)&v81, *v69);
      v9 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
      v10 = *(_QWORD *)(a1 + 16);
      v71 = 257;
      v11 = *(__int64 ***)(v10 + 80);
      if ( v11 != *(__int64 ***)(v9 + 8) )
        break;
      v13 = v9;
LABEL_12:
      if ( (unsigned int)(v76 - 24) > 1 )
      {
        v73 = 257;
        v14 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v10 + 80), 8, 0);
        v13 = sub_929C50(&v81, (_BYTE *)v13, v14, (__int64)v72, 0, 0);
        v10 = *(_QWORD *)(a1 + 16);
      }
      v71 = 257;
      v15 = *(_QWORD *)(v10 + 96);
      v16 = *(_QWORD *)(v13 + 8);
      if ( v15 == v16 )
      {
        v19 = v13;
        goto LABEL_22;
      }
      v17 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v89 + 120LL);
      if ( v17 == sub_920130 )
      {
        if ( *(_BYTE *)v13 > 0x15u )
          goto LABEL_37;
        v64 = (__int64 **)v15;
        if ( (unsigned __int8)sub_AC4810(0x30u) )
          v18 = sub_ADAB70(48, v13, v64, 0);
        else
          v18 = sub_AA93C0(0x30u, v13, (__int64)v64);
        v15 = (__int64)v64;
        v19 = v18;
      }
      else
      {
        v66 = v15;
        v49 = v17(v89, 48u, (_BYTE *)v13, v15);
        v15 = v66;
        v19 = v49;
      }
      if ( !v19 )
      {
LABEL_37:
        v73 = 257;
        v19 = sub_B51D30(48, v13, v15, (__int64)v72, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v39 = v92;
          if ( v91 )
            sub_B99FD0(v19, 3u, v91);
          sub_B45150(v19, v39);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v90 + 16LL))(
          v90,
          v19,
          v70,
          v86,
          v87);
        v40 = v81;
        v41 = &v81[4 * v82];
        if ( v81 != v41 )
        {
          do
          {
            v42 = *((_QWORD *)v40 + 1);
            v43 = *v40;
            v40 += 4;
            sub_B99FD0(v19, v43, v42);
          }
          while ( v41 != v40 );
        }
      }
      v16 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 96LL);
LABEL_22:
      v71 = 257;
      v20 = sub_AA4E30(v85);
      v21 = sub_AE5020(v20, v16);
      HIBYTE(v22) = HIBYTE(v68);
      v73 = 257;
      LOBYTE(v22) = v21;
      v68 = v22;
      v23 = sub_BD2C40(80, unk_3F10A14);
      v24 = (__int64)v23;
      if ( v23 )
        sub_B4D190((__int64)v23, v16, v19, (__int64)v72, 0, v68, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v90 + 16LL))(
        v90,
        v24,
        v70,
        v86,
        v87);
      v25 = v81;
      v26 = &v81[4 * v82];
      if ( v81 != v26 )
      {
        do
        {
          v27 = *((_QWORD *)v25 + 1);
          v28 = *v25;
          v25 += 4;
          sub_B99FD0(v24, v28, v27);
        }
        while ( v26 != v25 );
      }
      v29 = -1;
      v30 = sub_B2BEC0(*(_QWORD *)(a1 + 8));
      v31 = sub_9208B0(v30, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL));
      v72[1] = v32;
      v72[0] = (unsigned __int64)(v31 + 7) >> 3;
      LODWORD(v33) = sub_CA1930(v72);
      if ( (_DWORD)v33 )
      {
        _BitScanReverse64((unsigned __int64 *)&v33, (unsigned int)v33);
        v29 = 63 - (v33 ^ 0x3F);
      }
      v65 = *(_QWORD *)(a1 + 24);
      v34 = sub_BCB2B0(v88);
      LOBYTE(v35) = v29;
      HIBYTE(v35) = 1;
      if ( **(_BYTE **)(v65 + 8) )
        v36 = (__int64)sub_2465B30((__int64 *)v65, v24, (__int64)&v81, v34, 1);
      else
        v36 = sub_2463FC0(v65, v24, &v81, v35);
      LOBYTE(v1) = v29;
      v5 = 238;
      v37 = v1;
      BYTE1(v37) = 1;
      v1 = v37;
      v38 = v67;
      LOBYTE(v38) = v29;
      BYTE1(v38) = 1;
      v67 = v38;
      sub_B343C0((__int64)&v81, 0xEEu, v36, v1, *(_QWORD *)(a1 + 184), v38, v63, 0, 0, 0, 0, 0);
      nullsub_61();
      v93 = &unk_49DA100;
      nullsub_63();
      if ( v81 != (unsigned int *)&v83 )
        _libc_free((unsigned __int64)v81);
      if ( v62 == ++v69 )
        goto LABEL_33;
    }
    v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v89 + 120LL);
    if ( v12 == sub_920130 )
    {
      if ( *(_BYTE *)v9 > 0x15u )
        goto LABEL_44;
      if ( (unsigned __int8)sub_AC4810(0x2Fu) )
        v13 = sub_ADAB70(47, v9, v11, 0);
      else
        v13 = sub_AA93C0(0x2Fu, v9, (__int64)v11);
    }
    else
    {
      v13 = v12(v89, 47u, (_BYTE *)v9, (__int64)v11);
    }
    if ( v13 )
    {
LABEL_11:
      v10 = *(_QWORD *)(a1 + 16);
      goto LABEL_12;
    }
LABEL_44:
    v73 = 257;
    v13 = sub_B51D30(47, v9, (__int64)v11, (__int64)v72, 0, 0);
    if ( (unsigned __int8)sub_920620(v13) )
    {
      v44 = v92;
      if ( v91 )
        sub_B99FD0(v13, 3u, v91);
      sub_B45150(v13, v44);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v90 + 16LL))(
      v90,
      v13,
      v70,
      v86,
      v87);
    v45 = v81;
    v46 = &v81[4 * v82];
    if ( v81 != v46 )
    {
      do
      {
        v47 = *((_QWORD *)v45 + 1);
        v48 = *v45;
        v45 += 4;
        sub_B99FD0(v13, v48, v47);
      }
      while ( v46 != v45 );
    }
    goto LABEL_11;
  }
LABEL_33:
  if ( v74 != v75 )
  {
    v5 = v75[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v74);
  }
  sub_F94A20(v79, v5);
}
