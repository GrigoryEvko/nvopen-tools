// Function: sub_1CF8600
// Address: 0x1cf8600
//
void __fastcall sub_1CF8600(__int64 *a1)
{
  unsigned __int8 **v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned int v5; // r14d
  __int64 *v6; // r14
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // r12
  _QWORD *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // r13
  __int64 *v17; // r15
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 *v25; // r8
  __int64 v26; // r9
  __int64 *v27; // r8
  __int64 v28; // rax
  __int64 v29; // r12
  unsigned __int8 *v30; // rdx
  __int64 *v31; // r13
  __int64 v32; // rbx
  bool v33; // al
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 *v36; // r14
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // [rsp+8h] [rbp-3A8h]
  __int64 v42; // [rsp+10h] [rbp-3A0h]
  unsigned __int8 *v43; // [rsp+38h] [rbp-378h] BYREF
  __int64 v44[2]; // [rsp+40h] [rbp-370h] BYREF
  __int64 v45[2]; // [rsp+50h] [rbp-360h] BYREF
  __int16 v46; // [rsp+60h] [rbp-350h]
  __int64 v47; // [rsp+70h] [rbp-340h] BYREF
  __int64 v48; // [rsp+78h] [rbp-338h]
  __int64 v49; // [rsp+80h] [rbp-330h]
  __int64 v50; // [rsp+88h] [rbp-328h]
  unsigned __int8 *v51[2]; // [rsp+90h] [rbp-320h] BYREF
  _QWORD *v52; // [rsp+A0h] [rbp-310h]
  __int64 v53; // [rsp+A8h] [rbp-308h]
  __int64 v54; // [rsp+B0h] [rbp-300h]
  int v55; // [rsp+B8h] [rbp-2F8h]
  __int64 v56; // [rsp+C0h] [rbp-2F0h]
  __int64 v57; // [rsp+C8h] [rbp-2E8h]
  unsigned __int8 *v58; // [rsp+E0h] [rbp-2D0h] BYREF
  __int64 v59; // [rsp+E8h] [rbp-2C8h]
  _QWORD v60[3]; // [rsp+F0h] [rbp-2C0h] BYREF
  int v61; // [rsp+108h] [rbp-2A8h]
  __int64 v62; // [rsp+110h] [rbp-2A0h]
  __int64 v63; // [rsp+118h] [rbp-298h]
  _BYTE *v64; // [rsp+170h] [rbp-240h] BYREF
  __int64 v65; // [rsp+178h] [rbp-238h]
  _BYTE v66[560]; // [rsp+180h] [rbp-230h] BYREF

  v1 = &v58;
  v44[1] = (__int64)&v64;
  v2 = a1[2];
  v49 = 0;
  v3 = *a1;
  v50 = 0;
  v44[0] = (__int64)&v47;
  v4 = *(_QWORD *)(v2 + 16);
  v64 = v66;
  v65 = 0x2000000000LL;
  v47 = 0;
  v48 = 0;
  sub_1CF8370(v44, v3, v4);
  v5 = v65;
  if ( (_DWORD)v65 )
  {
    while ( 1 )
    {
      v14 = v5--;
      v15 = (unsigned __int64)&v64[16 * v14 - 16];
      v16 = *(__int64 **)v15;
      v12 = *(_QWORD *)(v15 + 8);
      LODWORD(v65) = v5;
      v17 = sub_1648700((__int64)v16);
      v18 = *((_BYTE *)v17 + 16);
      if ( v18 <= 0x17u )
        goto LABEL_17;
      if ( (unsigned __int8)(v18 - 71) <= 1u )
        break;
      if ( v18 == 56 )
      {
        v22 = sub_16498A0((__int64)v17);
        v51[0] = 0;
        v53 = v22;
        v54 = 0;
        v55 = 0;
        v56 = 0;
        v57 = 0;
        v51[1] = (unsigned __int8 *)v17[5];
        v52 = v17 + 3;
        v23 = (unsigned __int8 *)v17[6];
        v58 = v23;
        if ( v23 )
        {
          sub_1623A60((__int64)v1, (__int64)v23, 2);
          if ( v51[0] )
            sub_161E7C0((__int64)v51, (__int64)v51[0]);
          v51[0] = v58;
          if ( v58 )
            sub_1623210((__int64)v1, v58, (__int64)v51);
        }
        v24 = v17[7];
        v58 = (unsigned __int8 *)v60;
        v42 = v24;
        v59 = 0x1000000000LL;
        v25 = &v17[3 * (1LL - (*((_DWORD *)v17 + 5) & 0xFFFFFFF))];
        if ( v17 != v25 )
        {
          v26 = *v25;
          v27 = v25 + 3;
          v41 = v12;
          v28 = 0;
          v29 = (__int64)v1;
          v30 = (unsigned __int8 *)v60;
          v31 = v27;
          v32 = v26;
          while ( 1 )
          {
            *(_QWORD *)&v30[8 * v28] = v32;
            v28 = (unsigned int)(v59 + 1);
            LODWORD(v59) = v59 + 1;
            if ( v17 == v31 )
              break;
            v32 = *v31;
            if ( HIDWORD(v59) <= (unsigned int)v28 )
            {
              sub_16CD150(v29, v60, 0, 8, (int)v27, v26);
              v28 = (unsigned int)v59;
            }
            v30 = v58;
            v31 += 3;
          }
          v1 = (unsigned __int8 **)v29;
          v12 = v41;
        }
        v33 = sub_15FA300((__int64)v17);
        v46 = 257;
        if ( v33 )
          v34 = sub_128B460((__int64 *)v51, v42, (_BYTE *)v12, (__int64 **)v58, (unsigned int)v59, (__int64)v45);
        else
          v34 = sub_1BBF860((__int64 *)v51, v42, (_BYTE *)v12, (__int64 **)v58, (unsigned int)v59, v45);
        sub_1CF8370(v44, (__int64)v17, v34);
        if ( v58 != (unsigned __int8 *)v60 )
          _libc_free((unsigned __int64)v58);
        if ( v51[0] )
          sub_161E7C0((__int64)v51, (__int64)v51[0]);
        goto LABEL_24;
      }
      if ( v18 == 55 )
      {
LABEL_15:
        if ( !v5 )
          goto LABEL_25;
      }
      else
      {
LABEL_17:
        if ( *v16 )
        {
          v19 = v16[1];
          v20 = v16[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v20 = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
        }
        *v16 = v12;
        if ( v12 )
        {
          v21 = *(_QWORD *)(v12 + 8);
          v16[1] = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
          v16[2] = (v12 + 8) | v16[2] & 3;
          *(_QWORD *)(v12 + 8) = v16;
        }
LABEL_24:
        v5 = v65;
        if ( !(_DWORD)v65 )
          goto LABEL_25;
      }
    }
    v6 = **(__int64 ***)(*v17 + 16);
    v7 = sub_16498A0((__int64)v17);
    v58 = 0;
    v60[1] = v7;
    v60[2] = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v59 = v17[5];
    v60[0] = v17 + 3;
    v8 = (unsigned __int8 *)v17[6];
    v51[0] = v8;
    if ( v8 )
    {
      sub_1623A60((__int64)v51, (__int64)v8, 2);
      if ( v58 )
        sub_161E7C0((__int64)v1, (__int64)v58);
      v58 = v51[0];
      if ( v51[0] )
        sub_1623210((__int64)v51, v51[0], (__int64)v1);
    }
    v46 = 257;
    v9 = sub_1646BA0(v6, 0);
    if ( v9 != *(_QWORD *)v12 )
    {
      if ( *(_BYTE *)(v12 + 16) <= 0x10u )
      {
        v10 = sub_15A4AD0((__int64 ***)v12, v9);
        v11 = v58;
        v12 = v10;
        goto LABEL_12;
      }
      LOWORD(v52) = 257;
      v35 = sub_15FDF90(v12, v9, (__int64)v51, 0);
      v12 = v35;
      if ( v59 )
      {
        v36 = (__int64 *)v60[0];
        sub_157E9D0(v59 + 40, v35);
        v37 = *(_QWORD *)(v12 + 24);
        v38 = *v36;
        *(_QWORD *)(v12 + 32) = v36;
        v38 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v38 | v37 & 7;
        *(_QWORD *)(v38 + 8) = v12 + 24;
        *v36 = *v36 & 7 | (v12 + 24);
      }
      sub_164B780(v12, v45);
      if ( !v58 )
        goto LABEL_14;
      v43 = v58;
      sub_1623A60((__int64)&v43, (__int64)v58, 2);
      v39 = *(_QWORD *)(v12 + 48);
      if ( v39 )
        sub_161E7C0(v12 + 48, v39);
      v40 = v43;
      *(_QWORD *)(v12 + 48) = v43;
      if ( v40 )
        sub_1623210((__int64)&v43, v40, v12 + 48);
    }
    v11 = v58;
LABEL_12:
    if ( v11 )
      sub_161E7C0((__int64)v1, (__int64)v11);
LABEL_14:
    v13 = sub_1648700((__int64)v16);
    sub_1CF8370(v44, (__int64)v13, v12);
    v5 = v65;
    goto LABEL_15;
  }
LABEL_25:
  j___libc_free_0(v48);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
}
