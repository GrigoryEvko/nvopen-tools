// Function: sub_246B6A0
// Address: 0x246b6a0
//
void __fastcall sub_246B6A0(__int64 a1)
{
  unsigned int v1; // r15d
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD **v19; // rax
  __int64 v20; // r12
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 **v24; // r14
  __int64 **v25; // rax
  __int64 (__fastcall *v26)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v27; // r13
  __int64 (__fastcall *v28)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v29; // r14
  __int64 v30; // rax
  char v31; // al
  char v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // r12
  unsigned int *v35; // r14
  unsigned int *v36; // r13
  __int64 v37; // rdx
  unsigned int v38; // esi
  char v39; // r13
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rcx
  unsigned __int16 v45; // ax
  __int64 v46; // rdx
  unsigned int v47; // eax
  unsigned int v48; // eax
  int v49; // r12d
  unsigned int *v50; // r13
  unsigned int *v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // esi
  int v54; // r12d
  unsigned int *v55; // r14
  unsigned int *v56; // r12
  __int64 v57; // rdx
  unsigned int v58; // esi
  _QWORD **v59; // [rsp+28h] [rbp-228h]
  __int64 v60; // [rsp+30h] [rbp-220h]
  unsigned __int16 v61; // [rsp+42h] [rbp-20Eh]
  __int64 **v62; // [rsp+50h] [rbp-200h]
  __int64 v63; // [rsp+58h] [rbp-1F8h]
  __int64 v64; // [rsp+58h] [rbp-1F8h]
  _QWORD **v65; // [rsp+68h] [rbp-1E8h]
  char v66[32]; // [rsp+70h] [rbp-1E0h] BYREF
  __int16 v67; // [rsp+90h] [rbp-1C0h]
  _BYTE v68[32]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v69; // [rsp+C0h] [rbp-190h]
  _QWORD v70[4]; // [rsp+D0h] [rbp-180h] BYREF
  __int16 v71; // [rsp+F0h] [rbp-160h]
  unsigned int *v72[9]; // [rsp+100h] [rbp-150h] BYREF
  _QWORD *v73; // [rsp+148h] [rbp-108h]
  unsigned int *v74; // [rsp+190h] [rbp-C0h] BYREF
  int v75; // [rsp+198h] [rbp-B8h]
  char v76; // [rsp+1A0h] [rbp-B0h] BYREF
  __int16 v77; // [rsp+1B0h] [rbp-A0h]
  __int64 v78; // [rsp+1C0h] [rbp-90h]
  __int64 v79; // [rsp+1C8h] [rbp-88h]
  __int64 v80; // [rsp+1D0h] [rbp-80h]
  _QWORD *v81; // [rsp+1D8h] [rbp-78h]
  __int64 v82; // [rsp+1E0h] [rbp-70h]
  __int64 v83; // [rsp+1E8h] [rbp-68h]
  __int64 v84; // [rsp+1F0h] [rbp-60h]
  int v85; // [rsp+1F8h] [rbp-58h]
  void *v86; // [rsp+210h] [rbp-40h]

  sub_23D0AB0((__int64)v72, *(_QWORD *)(*(_QWORD *)(a1 + 24) + 480LL), 0, 0, 0);
  v2 = *(_QWORD *)(a1 + 16);
  v77 = 257;
  v3 = *(_QWORD *)(v2 + 152);
  v4 = sub_BCB2E0(v73);
  v5 = sub_A82CA0(v72, v4, v3, 0, 0, (__int64)&v74);
  v6 = *(_DWORD *)(a1 + 40);
  v60 = v5;
  *(_QWORD *)(a1 + 192) = v5;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v77 = 257;
    v8 = (__int64 *)sub_BCB2B0(*(_QWORD **)(v7 + 72));
    v9 = sub_23DEB90((__int64 *)v72, v8, v60, (__int64)&v74);
    v10 = (unsigned __int8)byte_4FE8EA8;
    *(_QWORD *)(a1 + 184) = v9;
    *(_WORD *)(v9 + 2) = v10 | *(_WORD *)(v9 + 2) & 0xFFC0;
    LODWORD(v9) = (unsigned __int8)v10;
    BYTE1(v9) = 1;
    v11 = v9;
    v12 = sub_BCB2B0(v73);
    v13 = sub_AD6530(v12, v10);
    sub_B34240((__int64)v72, *(_QWORD *)(a1 + 184), v13, v60, v11, 0, 0, 0, 0);
    v77 = 257;
    HIDWORD(v70[0]) = 0;
    v14 = sub_BCB2E0(v73);
    v15 = sub_ACD640(v14, 800, 0);
    v4 = 238;
    v16 = sub_B33C40((__int64)v72, 0x16Eu, v60, v15, LODWORD(v70[0]), (__int64)&v74);
    v17 = (unsigned __int8)byte_4FE8EA8;
    v18 = (unsigned __int8)byte_4FE8EA8;
    BYTE1(v17) = 1;
    BYTE1(v18) = 1;
    sub_B343C0(
      (__int64)v72,
      0xEEu,
      *(_QWORD *)(a1 + 184),
      v17,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 136LL),
      v18,
      v16,
      0,
      0,
      0,
      0,
      0);
    v19 = *(_QWORD ***)(a1 + 32);
    v59 = &v19[*(unsigned int *)(a1 + 40)];
    if ( v59 != v19 )
    {
      v65 = *(_QWORD ***)(a1 + 32);
      while ( 1 )
      {
        v20 = (__int64)*v65;
        sub_2468350((__int64)&v74, *v65);
        v21 = *(_QWORD *)(v20 - 32LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF));
        v63 = sub_BCE3C0(*(__int64 **)(*(_QWORD *)(a1 + 16) + 72LL), 0);
        v22 = *(_QWORD *)(a1 + 16);
        v69 = 257;
        v62 = (__int64 **)sub_BCE3C0(*(__int64 **)(v22 + 72), 0);
        v23 = *(_QWORD *)(a1 + 16);
        v67 = 257;
        v24 = *(__int64 ***)(v23 + 80);
        v25 = *(__int64 ***)(v21 + 8);
        if ( v24 != v25 )
          break;
        v27 = v21;
LABEL_12:
        if ( v62 == v25 )
        {
          v29 = v27;
        }
        else
        {
          v28 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v82 + 120LL);
          if ( v28 == sub_920130 )
          {
            if ( *(_BYTE *)v27 > 0x15u )
              goto LABEL_31;
            if ( (unsigned __int8)sub_AC4810(0x30u) )
              v29 = sub_ADAB70(48, v27, v62, 0);
            else
              v29 = sub_AA93C0(0x30u, v27, (__int64)v62);
          }
          else
          {
            v29 = v28(v82, 48u, (_BYTE *)v27, (__int64)v62);
          }
          if ( !v29 )
          {
LABEL_31:
            v71 = 257;
            v29 = sub_B51D30(48, v27, (__int64)v62, (__int64)v70, 0, 0);
            if ( (unsigned __int8)sub_920620(v29) )
            {
              v49 = v85;
              if ( v84 )
                sub_B99FD0(v29, 3u, v84);
              sub_B45150(v29, v49);
            }
            (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v83 + 16LL))(
              v83,
              v29,
              v68,
              v79,
              v80);
            v50 = v74;
            v51 = &v74[4 * v75];
            if ( v74 != v51 )
            {
              do
              {
                v52 = *((_QWORD *)v50 + 1);
                v53 = *v50;
                v50 += 4;
                sub_B99FD0(v29, v53, v52);
              }
              while ( v51 != v50 );
            }
          }
        }
        v69 = 257;
        v30 = sub_AA4E30(v78);
        v31 = sub_AE5020(v30, v63);
        v71 = 257;
        v32 = v31;
        v33 = sub_BD2C40(80, unk_3F10A14);
        v34 = (__int64)v33;
        if ( v33 )
          sub_B4D190((__int64)v33, v63, v29, (__int64)v70, 0, v32, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v83 + 16LL))(
          v83,
          v34,
          v68,
          v79,
          v80);
        v35 = v74;
        v36 = &v74[4 * v75];
        if ( v74 != v36 )
        {
          do
          {
            v37 = *((_QWORD *)v35 + 1);
            v38 = *v35;
            v35 += 4;
            sub_B99FD0(v34, v38, v37);
          }
          while ( v36 != v35 );
        }
        v39 = -1;
        v40 = sub_B2BEC0(*(_QWORD *)(a1 + 8));
        v41 = sub_9208B0(v40, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL));
        v70[1] = v42;
        v70[0] = (unsigned __int64)(v41 + 7) >> 3;
        LODWORD(v43) = sub_CA1930(v70);
        if ( (_DWORD)v43 )
        {
          _BitScanReverse64((unsigned __int64 *)&v43, (unsigned int)v43);
          v39 = 63 - (v43 ^ 0x3F);
        }
        v64 = *(_QWORD *)(a1 + 24);
        v44 = sub_BCB2B0(v81);
        LOBYTE(v45) = v39;
        HIBYTE(v45) = 1;
        if ( **(_BYTE **)(v64 + 8) )
          v46 = (__int64)sub_2465B30((__int64 *)v64, v34, (__int64)&v74, v44, 1);
        else
          v46 = sub_2463FC0(v64, v34, &v74, v45);
        LOBYTE(v1) = v39;
        v4 = 238;
        v47 = v1;
        BYTE1(v47) = 1;
        v1 = v47;
        v48 = v61;
        LOBYTE(v48) = v39;
        BYTE1(v48) = 1;
        v61 = v48;
        sub_B343C0((__int64)&v74, 0xEEu, v46, v1, *(_QWORD *)(a1 + 184), v48, v60, 0, 0, 0, 0, 0);
        nullsub_61();
        v86 = &unk_49DA100;
        nullsub_63();
        if ( v74 != (unsigned int *)&v76 )
          _libc_free((unsigned __int64)v74);
        if ( v59 == ++v65 )
          goto LABEL_2;
      }
      v26 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v82 + 120LL);
      if ( v26 == sub_920130 )
      {
        if ( *(_BYTE *)v21 > 0x15u )
          goto LABEL_38;
        if ( (unsigned __int8)sub_AC4810(0x2Fu) )
          v27 = sub_ADAB70(47, v21, v24, 0);
        else
          v27 = sub_AA93C0(0x2Fu, v21, (__int64)v24);
      }
      else
      {
        v27 = v26(v82, 47u, (_BYTE *)v21, (__int64)v24);
      }
      if ( v27 )
      {
LABEL_11:
        v25 = *(__int64 ***)(v27 + 8);
        goto LABEL_12;
      }
LABEL_38:
      v71 = 257;
      v27 = sub_B51D30(47, v21, (__int64)v24, (__int64)v70, 0, 0);
      if ( (unsigned __int8)sub_920620(v27) )
      {
        v54 = v85;
        if ( v84 )
          sub_B99FD0(v27, 3u, v84);
        sub_B45150(v27, v54);
      }
      (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v83 + 16LL))(
        v83,
        v27,
        v66,
        v79,
        v80);
      v55 = v74;
      v56 = &v74[4 * v75];
      if ( v74 != v56 )
      {
        do
        {
          v57 = *((_QWORD *)v55 + 1);
          v58 = *v55;
          v55 += 4;
          sub_B99FD0(v27, v58, v57);
        }
        while ( v56 != v55 );
      }
      goto LABEL_11;
    }
  }
LABEL_2:
  sub_F94A20(v72, v4);
}
