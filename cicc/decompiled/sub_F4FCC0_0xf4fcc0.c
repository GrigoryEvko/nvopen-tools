// Function: sub_F4FCC0
// Address: 0xf4fcc0
//
void __fastcall sub_F4FCC0(_BYTE *a1, __int64 a2, char a3, char a4)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v9; // r8
  _BYTE *v10; // r15
  _BYTE *v11; // rbx
  unsigned int v12; // r13d
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 v15; // rax
  size_t v16; // rdx
  __int64 v17; // r11
  __int64 v18; // r10
  __int64 *v19; // r8
  char *v20; // rsi
  size_t v21; // rax
  __int64 v22; // rax
  size_t v23; // rdx
  __int64 v24; // r11
  __int64 v25; // r10
  __int64 *v26; // r8
  size_t v27; // rax
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned __int8 v55; // al
  __int64 v56; // rax
  unsigned __int8 v57; // al
  __int64 v58; // rax
  __int64 v59; // [rsp+0h] [rbp-100h]
  __int64 v60; // [rsp+8h] [rbp-F8h]
  __int64 v61; // [rsp+8h] [rbp-F8h]
  __int64 v62; // [rsp+10h] [rbp-F0h]
  int v63; // [rsp+10h] [rbp-F0h]
  __int64 v64; // [rsp+18h] [rbp-E8h]
  __int64 *v65; // [rsp+18h] [rbp-E8h]
  __int64 v66; // [rsp+18h] [rbp-E8h]
  __int64 *v67; // [rsp+18h] [rbp-E8h]
  __int64 v68; // [rsp+18h] [rbp-E8h]
  __int64 v69; // [rsp+18h] [rbp-E8h]
  __int64 v70; // [rsp+18h] [rbp-E8h]
  __int64 v71; // [rsp+20h] [rbp-E0h]
  char *v72; // [rsp+20h] [rbp-E0h]
  __int64 v73; // [rsp+20h] [rbp-E0h]
  char *v74; // [rsp+20h] [rbp-E0h]
  __int64 v75; // [rsp+20h] [rbp-E0h]
  __int64 v76; // [rsp+20h] [rbp-E0h]
  __int64 v77; // [rsp+20h] [rbp-E0h]
  __int64 v78; // [rsp+20h] [rbp-E0h]
  __int64 v81; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v82[3]; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v83; // [rsp+58h] [rbp-A8h]
  __int64 v84[3]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v85; // [rsp+78h] [rbp-88h]
  _BYTE *v86; // [rsp+80h] [rbp-80h] BYREF
  __int64 v87; // [rsp+88h] [rbp-78h]
  _BYTE v88[112]; // [rsp+90h] [rbp-70h] BYREF

  v81 = a2;
  v5 = (__int64)&v86;
  v86 = v88;
  v87 = 0x400000000LL;
  sub_B9A9D0((__int64)a1, (__int64)&v86);
  v9 = (__int64)v86;
  v10 = v86;
  v11 = &v86[16 * (unsigned int)v87];
  if ( v11 != v86 )
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)v10;
      if ( !*(_DWORD *)v10 )
        BUG();
      v13 = 0;
      if ( (*(_BYTE *)(v81 + 7) & 0x20) != 0 )
      {
        v5 = v12;
        v13 = sub_B91C10(v81, v12);
      }
      v14 = *((_QWORD *)v10 + 1);
      switch ( v12 )
      {
        case 1u:
          if ( !a3 )
            goto LABEL_15;
          v51 = *((_QWORD *)v10 + 1);
          v10 += 16;
          v52 = sub_E01DF0(v13, v51, v6, v7, v9, v8);
          v5 = 1;
          sub_B99FD0((__int64)a1, 1u, v52);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 2u:
          if ( a4 == 1 || !a3 )
            goto LABEL_15;
          v49 = *((_QWORD *)v10 + 1);
          v10 += 16;
          v50 = sub_B9DD00(v49, v13, (__int64)a1, v81);
          v5 = 2;
          sub_B99FD0((__int64)a1, 2u, v50);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 3u:
          if ( a4 )
            goto LABEL_15;
          v10 += 16;
          v48 = sub_B916B0(v13, v14, v6, v7, v9);
          v5 = 3;
          sub_B99FD0((__int64)a1, 3u, v48);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 4u:
          if ( a4 )
            goto LABEL_15;
          if ( !a3 && (a1[7] & 0x20) != 0 )
          {
            v5 = 29;
            v70 = v13;
            v78 = *((_QWORD *)v10 + 1);
            v46 = sub_B91C10((__int64)a1, 29);
            v14 = v78;
            v13 = v70;
            if ( v46 )
              goto LABEL_15;
          }
          v10 += 16;
          v47 = sub_B9DD80(v13, v14);
          v5 = 4;
          sub_B99FD0((__int64)a1, 4u, v47);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 6u:
          if ( !a3 )
            goto LABEL_15;
          v5 = 6;
          v10 += 16;
          sub_B99FD0((__int64)a1, 6u, v13);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 7u:
          if ( !a3 )
            goto LABEL_15;
          v44 = (__int64 *)*((_QWORD *)v10 + 1);
          v10 += 16;
          v45 = sub_BA6CD0(v13, v44);
          v5 = 7;
          sub_B99FD0((__int64)a1, 7u, v45);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 8u:
        case 0xAu:
          if ( !a3 )
            goto LABEL_15;
          v36 = *((_QWORD *)v10 + 1);
          v10 += 16;
          v37 = sub_BA74A0(v13, v36, v6, v7);
          v5 = v12;
          sub_B99FD0((__int64)a1, v12, v37);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 9u:
          if ( a4 )
            goto LABEL_15;
          v5 = 9;
          v10 += 16;
          sub_B99FD0((__int64)a1, 9u, v13);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0xBu:
          if ( a4 )
            goto LABEL_15;
          if ( !a3 && (a1[7] & 0x20) != 0 )
          {
            v5 = 29;
            v76 = v13;
            v39 = sub_B91C10((__int64)a1, 29);
            v13 = v76;
            if ( v39 )
              goto LABEL_15;
          }
          v5 = 11;
          v10 += 16;
          sub_B99FD0((__int64)a1, 0xBu, v13);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0xCu:
        case 0xDu:
          if ( a4 == 1 || !a3 )
            goto LABEL_15;
          v34 = *((_QWORD *)v10 + 1);
          v10 += 16;
          v35 = sub_B918F0(v13, v34);
          v5 = v12;
          sub_B99FD0((__int64)a1, v12, v35);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x10u:
        case 0x1Bu:
        case 0x28u:
          goto LABEL_15;
        case 0x11u:
          if ( a4 )
            goto LABEL_15;
          if ( !a3 && (a1[7] & 0x20) != 0 )
          {
            v5 = 29;
            v69 = v13;
            v77 = *((_QWORD *)v10 + 1);
            v42 = sub_B91C10((__int64)a1, 29);
            v14 = v77;
            v13 = v69;
            if ( v42 )
              goto LABEL_15;
          }
          v10 += 16;
          v43 = sub_B918F0(v13, v14);
          v5 = 17;
          sub_B99FD0((__int64)a1, 0x11u, v43);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x19u:
          if ( !a3 )
            goto LABEL_15;
          v10 += 16;
          v41 = sub_9B8B30((__int64)a1, v81);
          v5 = 25;
          sub_B99FD0((__int64)a1, 0x19u, v41);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x1Du:
          if ( a4 == 1 || !a3 )
            goto LABEL_15;
          v5 = 29;
          v10 += 16;
          sub_B99FD0((__int64)a1, 0x1Du, v13);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x22u:
          if ( a4 )
            goto LABEL_15;
          v10 += 16;
          v40 = sub_1039CA0(v14, v13);
          v5 = 34;
          sub_B99FD0((__int64)a1, 0x22u, v40);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x23u:
          if ( a4 )
            goto LABEL_15;
          v10 += 16;
          v38 = sub_1039CB0(v14, v13);
          v5 = 35;
          sub_B99FD0((__int64)a1, 0x23u, v38);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x26u:
          if ( a4 )
            goto LABEL_15;
          v5 = (__int64)&v81;
          v10 += 16;
          sub_AE9860((__int64)a1, (__int64)&v81, 1);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        case 0x29u:
          if ( !a3 )
            goto LABEL_15;
          v53 = *((_QWORD *)v10 + 1);
          v10 += 16;
          v54 = sub_B9E990(v13, v53);
          v5 = 41;
          sub_B99FD0((__int64)a1, 0x29u, v54);
          if ( v11 == v10 )
            goto LABEL_16;
          continue;
        default:
          v64 = v13;
          v71 = *((_QWORD *)v10 + 1);
          v15 = sub_BD5C60((__int64)a1);
          v16 = 0;
          v17 = v71;
          v18 = v64;
          v19 = (__int64 *)v15;
          v20 = off_4C5D0D8[0];
          if ( off_4C5D0D8[0] )
          {
            v60 = v64;
            v62 = v71;
            v65 = (__int64 *)v15;
            v72 = off_4C5D0D8[0];
            v21 = strlen(off_4C5D0D8[0]);
            v18 = v60;
            v17 = v62;
            v19 = v65;
            v20 = v72;
            v16 = v21;
          }
          v66 = v18;
          v73 = v17;
          v63 = sub_B6ED60(v19, v20, v16);
          v22 = sub_BD5C60((__int64)a1);
          v23 = 0;
          v24 = v73;
          v25 = v66;
          v26 = (__int64 *)v22;
          v5 = (__int64)off_4C5D0D0[0];
          if ( off_4C5D0D0[0] )
          {
            v59 = v66;
            v61 = v73;
            v67 = (__int64 *)v22;
            v74 = off_4C5D0D0[0];
            v27 = strlen(off_4C5D0D0[0]);
            v25 = v59;
            v24 = v61;
            v26 = v67;
            v5 = (__int64)v74;
            v23 = v27;
          }
          v75 = v24;
          v68 = v25;
          v28 = sub_B6ED60(v26, (const void *)v5, v23);
          if ( !v75 || !v68 )
            goto LABEL_14;
          if ( v12 == v63 )
          {
            if ( v75 != v68 )
              goto LABEL_14;
          }
          else if ( v12 != v28
                 || ((v55 = *(_BYTE *)(v68 - 16), (v55 & 2) != 0)
                   ? (v56 = *(_QWORD *)(v68 - 32))
                   : (v56 = v68 - 8LL * ((v55 >> 2) & 0xF) - 16),
                     (v6 = *(_QWORD *)(v56 + 8), v57 = *(_BYTE *)(v75 - 16), (v57 & 2) == 0)
                   ? (v58 = v75 - 8LL * ((v57 >> 2) & 0xF) - 16)
                   : (v58 = *(_QWORD *)(v75 - 32)),
                     v6 != *(_QWORD *)(v58 + 8)) )
          {
LABEL_14:
            v5 = v12;
            sub_B99FD0((__int64)a1, v12, 0);
          }
LABEL_15:
          v10 += 16;
          if ( v11 == v10 )
            goto LABEL_16;
          break;
      }
    }
  }
LABEL_16:
  if ( (*(_BYTE *)(v81 + 7) & 0x20) == 0 )
    goto LABEL_92;
  v5 = 16;
  v29 = sub_B91C10(v81, 16);
  if ( v29 && (unsigned __int8)(*a1 - 61) <= 1u )
  {
    v5 = 16;
    sub_B99FD0((__int64)a1, 0x10u, v29);
  }
  if ( (*(_BYTE *)(v81 + 7) & 0x20) == 0 )
  {
LABEL_92:
    v31 = 0;
    if ( (a1[7] & 0x20) == 0 )
      goto LABEL_25;
    goto LABEL_22;
  }
  v30 = sub_B91C10(v81, 40);
  v31 = v30;
  if ( (a1[7] & 0x20) != 0 )
  {
LABEL_22:
    v5 = sub_B91C10((__int64)a1, 40);
    v30 = v31 | v5;
    goto LABEL_23;
  }
  v5 = 0;
LABEL_23:
  if ( v30 )
  {
    sub_B8DF90((__int64)v84, (_BYTE *)v5);
    sub_B8DF90((__int64)v82, (_BYTE *)v31);
    v32 = sub_BD5C60((__int64)a1);
    v33 = sub_B8D9F0(v32, v82, v84);
    sub_B99FD0((__int64)a1, 0x28u, v33);
    sub_C7D6A0(v82[1], 32LL * v83, 8);
    v5 = 32LL * v85;
    sub_C7D6A0(v84[1], v5, 8);
  }
LABEL_25:
  if ( v86 != v88 )
    _libc_free(v86, v5);
}
