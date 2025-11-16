// Function: sub_2C6D760
// Address: 0x2c6d760
//
__int64 __fastcall sub_2C6D760(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // r13
  void *v9; // rsi
  __int64 v10; // r13
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // r15
  unsigned __int8 *v19; // r15
  int v20; // eax
  unsigned __int8 *v21; // r13
  unsigned int v22; // edx
  unsigned __int8 **v23; // rax
  unsigned __int8 *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // eax
  int v30; // r8d
  __int64 *v31; // [rsp+8h] [rbp-A88h]
  __int64 v32; // [rsp+10h] [rbp-A80h]
  char v33; // [rsp+30h] [rbp-A60h]
  __int64 v34; // [rsp+38h] [rbp-A58h]
  __int64 v35; // [rsp+38h] [rbp-A58h]
  char v36; // [rsp+4Fh] [rbp-A41h] BYREF
  __int64 v37; // [rsp+50h] [rbp-A40h] BYREF
  char *v38; // [rsp+58h] [rbp-A38h]
  int v39; // [rsp+60h] [rbp-A30h]
  int v40; // [rsp+64h] [rbp-A2Ch]
  int v41; // [rsp+68h] [rbp-A28h]
  char v42; // [rsp+6Ch] [rbp-A24h]
  _QWORD v43[2]; // [rsp+70h] [rbp-A20h] BYREF
  __int64 v44; // [rsp+80h] [rbp-A10h] BYREF
  _BYTE *v45; // [rsp+88h] [rbp-A08h]
  __int64 v46; // [rsp+90h] [rbp-A00h]
  int v47; // [rsp+98h] [rbp-9F8h]
  char v48; // [rsp+9Ch] [rbp-9F4h]
  _BYTE v49[16]; // [rsp+A0h] [rbp-9F0h] BYREF
  __int64 v50; // [rsp+B0h] [rbp-9E0h] BYREF
  _BYTE *v51; // [rsp+B8h] [rbp-9D8h]
  __int64 v52; // [rsp+C0h] [rbp-9D0h]
  _BYTE v53[32]; // [rsp+C8h] [rbp-9C8h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-9A8h]
  __int64 v55; // [rsp+F0h] [rbp-9A0h]
  __int16 v56; // [rsp+F8h] [rbp-998h]
  __int64 v57; // [rsp+100h] [rbp-990h]
  void **v58; // [rsp+108h] [rbp-988h]
  _QWORD *v59; // [rsp+110h] [rbp-980h]
  __int64 v60; // [rsp+118h] [rbp-978h]
  int v61; // [rsp+120h] [rbp-970h]
  __int16 v62; // [rsp+124h] [rbp-96Ch]
  char v63; // [rsp+126h] [rbp-96Ah]
  __int64 v64; // [rsp+128h] [rbp-968h]
  __int64 v65; // [rsp+130h] [rbp-960h]
  void *v66; // [rsp+138h] [rbp-958h] BYREF
  _QWORD v67[2]; // [rsp+140h] [rbp-950h] BYREF
  __int64 v68; // [rsp+150h] [rbp-940h]
  __int64 v69; // [rsp+158h] [rbp-938h]
  __int64 v70; // [rsp+160h] [rbp-930h]
  __int64 v71; // [rsp+168h] [rbp-928h]
  int v72; // [rsp+170h] [rbp-920h]
  char v73; // [rsp+174h] [rbp-91Ch]
  _BYTE *v74; // [rsp+178h] [rbp-918h]
  __int64 v75; // [rsp+180h] [rbp-910h]
  _BYTE v76[2048]; // [rsp+188h] [rbp-908h] BYREF
  __int64 v77; // [rsp+988h] [rbp-108h]
  __int64 v78; // [rsp+990h] [rbp-100h]
  __int64 v79; // [rsp+998h] [rbp-F8h]
  unsigned int v80; // [rsp+9A0h] [rbp-F0h]
  __int64 v81; // [rsp+9A8h] [rbp-E8h]
  __int64 v82; // [rsp+9B0h] [rbp-E0h]
  __int64 v83; // [rsp+9B8h] [rbp-D8h]
  __int64 v84; // [rsp+9C0h] [rbp-D0h]
  _BYTE *v85; // [rsp+9C8h] [rbp-C8h]
  __int64 v86; // [rsp+9D0h] [rbp-C0h]
  _BYTE v87[184]; // [rsp+9D8h] [rbp-B8h] BYREF

  v6 = a1;
  v34 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v31 = (__int64 *)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v32 = sub_B2BEC0(a3);
  v33 = *a2;
  v50 = a3;
  v52 = 0x200000000LL;
  v62 = 512;
  v57 = sub_B2BE50(a3);
  v51 = v53;
  v66 = &unk_49DA100;
  v58 = &v66;
  v59 = v67;
  v56 = 0;
  v67[0] = &unk_49DA0B0;
  v60 = 0;
  v61 = 0;
  v63 = 7;
  v64 = 0;
  v65 = 0;
  v54 = 0;
  v55 = 0;
  v67[1] = v31;
  v68 = v7 + 8;
  v69 = v8 + 8;
  v70 = v34 + 8;
  v71 = v32;
  v72 = 0;
  v73 = v33;
  v74 = v76;
  v75 = 0x10000000000LL;
  v85 = v87;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v86 = 0x1000000000LL;
  if ( (_BYTE)qword_5010C88 || (sub_DFB180(v31, 1u), !(unsigned int)sub_DFB120((__int64)v31)) )
  {
    v9 = (void *)(a1 + 32);
    v10 = a1 + 80;
LABEL_3:
    *(_QWORD *)(v6 + 8) = v9;
    *(_QWORD *)(v6 + 16) = 0x100000002LL;
    *(_QWORD *)(v6 + 48) = 0;
    *(_QWORD *)(v6 + 56) = v10;
    *(_QWORD *)(v6 + 64) = 2;
    *(_DWORD *)(v6 + 72) = 0;
    *(_BYTE *)(v6 + 76) = 1;
    *(_DWORD *)(v6 + 24) = 0;
    *(_BYTE *)(v6 + 28) = 1;
    *(_QWORD *)(v6 + 32) = &qword_4F82400;
    *(_QWORD *)v6 = 1;
    goto LABEL_4;
  }
  v36 = 0;
  v37 = (__int64)&v50;
  v38 = &v36;
  v12 = *(_QWORD *)(v50 + 80);
  v35 = v50 + 72;
  if ( v12 != v50 + 72 )
  {
    do
    {
      if ( v12 )
      {
        v13 = v12 - 24;
        v14 = (unsigned int)(*(_DWORD *)(v12 + 20) + 1);
        v15 = *(_DWORD *)(v12 + 20) + 1;
      }
      else
      {
        v13 = 0;
        v14 = 0;
        v15 = 0;
      }
      if ( v15 < *(_DWORD *)(v68 + 32) )
      {
        if ( *(_QWORD *)(*(_QWORD *)(v68 + 24) + 8 * v14) )
        {
          v16 = *(_QWORD *)(v13 + 56);
          v17 = v13 + 48;
          while ( v17 != v16 )
          {
            v18 = v16;
            v16 = *(_QWORD *)(v16 + 8);
            v19 = (unsigned __int8 *)(v18 - 24);
            if ( !sub_B46AA0((__int64)v19) )
              sub_2C6BAF0(&v37, v19);
          }
        }
      }
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v35 != v12 );
    v6 = a1;
  }
LABEL_23:
  while ( 1 )
  {
    v20 = v75;
    if ( !(_DWORD)v75 )
      break;
    while ( 1 )
    {
      v21 = *(unsigned __int8 **)&v74[8 * v20 - 8];
      LODWORD(v75) = v20 - 1;
      if ( v80 )
      {
        v22 = (v80 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = (unsigned __int8 **)(v78 + 16LL * v22);
        v24 = *v23;
        if ( v21 == *v23 )
        {
LABEL_26:
          *v23 = (unsigned __int8 *)-8192LL;
          LODWORD(v79) = v79 - 1;
          ++HIDWORD(v79);
        }
        else
        {
          v29 = 1;
          while ( v24 != (unsigned __int8 *)-4096LL )
          {
            v30 = v29 + 1;
            v22 = (v80 - 1) & (v29 + v22);
            v23 = (unsigned __int8 **)(v78 + 16LL * v22);
            v24 = *v23;
            if ( v21 == *v23 )
              goto LABEL_26;
            v29 = v30;
          }
        }
      }
      if ( !v21 )
        break;
      if ( sub_F50EE0(v21, 0) )
      {
        sub_2C60650((__int64)&v50, (__int64)v21, v25, v26, v27, v28);
        goto LABEL_23;
      }
      sub_2C6BAF0(&v37, v21);
      v20 = v75;
      if ( !(_DWORD)v75 )
        goto LABEL_30;
    }
  }
LABEL_30:
  while ( (_DWORD)v86 )
    ;
  v9 = (void *)(v6 + 32);
  v10 = v6 + 80;
  if ( !v36 )
    goto LABEL_3;
  v38 = (char *)v43;
  v43[0] = &unk_4F82408;
  v39 = 2;
  v41 = 0;
  v42 = 1;
  v44 = 0;
  v45 = v49;
  v46 = 2;
  v47 = 0;
  v48 = 1;
  v40 = 1;
  v37 = 1;
  sub_C8CF70(v6, v9, 2, (__int64)v43, (__int64)&v37);
  sub_C8CF70(v6 + 48, (void *)(v6 + 80), 2, (__int64)v49, (__int64)&v44);
  if ( !v48 )
    _libc_free((unsigned __int64)v45);
  if ( !v42 )
    _libc_free((unsigned __int64)v38);
LABEL_4:
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
  sub_C7D6A0(v82, 8LL * (unsigned int)v84, 8);
  sub_C7D6A0(v78, 16LL * v80, 8);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  nullsub_61();
  v66 = &unk_49DA100;
  nullsub_63();
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return v6;
}
