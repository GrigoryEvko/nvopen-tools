// Function: sub_15311A0
// Address: 0x15311a0
//
void __fastcall sub_15311A0(__int64 ***a1)
{
  __int64 v2; // r13
  __int64 **v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  volatile signed __int32 *v7; // r8
  __int64 **v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  volatile signed __int32 *v12; // r8
  __int64 **v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  volatile signed __int32 *v17; // r8
  __int64 **v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  volatile signed __int32 *v22; // r8
  __int64 **v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  volatile signed __int32 *v27; // r8
  __int64 **v28; // rdi
  __int64 v29; // rax
  __int64 **v30; // rax
  __int64 **v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // r12
  unsigned int v35; // r9d
  unsigned int v36; // r15d
  unsigned int v37; // r12d
  __int64 **v38; // r15
  __int64 v39; // rax
  __int64 v40; // r11
  __int64 **v41; // r12
  unsigned int v42; // esi
  __int64 **v43; // rdx
  __int64 *v44; // r8
  __int64 v45; // r8
  __int64 v46; // rcx
  __int64 **v47; // r9
  int v48; // edx
  __int64 v49; // r11
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // r14
  __int64 *v53; // rdi
  unsigned int v54; // esi
  __int64 **v55; // rdx
  __int64 *v56; // r8
  __int64 v57; // r8
  __int64 v58; // rcx
  __int64 **v59; // r9
  int v60; // edx
  int v61; // edx
  unsigned int v62; // r9d
  int v63; // r15d
  __int64 v64; // rdx
  char *v65; // rax
  int v66; // edx
  __int64 v67; // [rsp+8h] [rbp-2B8h]
  __int64 v68; // [rsp+8h] [rbp-2B8h]
  __int64 v69; // [rsp+10h] [rbp-2B0h]
  __int64 v70; // [rsp+10h] [rbp-2B0h]
  int v71; // [rsp+10h] [rbp-2B0h]
  int v72; // [rsp+10h] [rbp-2B0h]
  unsigned int v73; // [rsp+10h] [rbp-2B0h]
  unsigned int v74; // [rsp+18h] [rbp-2A8h]
  unsigned int v75; // [rsp+1Ch] [rbp-2A4h]
  unsigned int v76; // [rsp+20h] [rbp-2A0h]
  unsigned int v77; // [rsp+24h] [rbp-29Ch]
  unsigned int v78; // [rsp+28h] [rbp-298h]
  unsigned int v79; // [rsp+2Ch] [rbp-294h]
  __int64 v80; // [rsp+38h] [rbp-288h]
  __int64 v81; // [rsp+48h] [rbp-278h]
  unsigned __int128 v82; // [rsp+60h] [rbp-260h] BYREF
  __m128i v83; // [rsp+70h] [rbp-250h] BYREF
  _BYTE *v84; // [rsp+80h] [rbp-240h] BYREF
  __int64 v85; // [rsp+88h] [rbp-238h]
  _BYTE v86[560]; // [rsp+90h] [rbp-230h] BYREF

  sub_1526BE0(*a1, 0x11u, 4u);
  v84 = v86;
  v85 = 0x4000000000LL;
  v80 = (__int64)(a1 + 3);
  v2 = sub_153EE90(a1 + 3);
  sub_1531130(&v82);
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 8;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = v2;
  v83.m128i_i8[8] = v83.m128i_i8[8] & 0xF0 | 2;
  sub_1525B40(v82, &v83);
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 0;
  sub_1525B40(v82, &v83);
  v3 = *a1;
  v4 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v4;
  v79 = sub_15271D0(v3, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  sub_1531130(&v83);
  v5 = v83.m128i_i64[1];
  v6 = v83.m128i_i64[0];
  v83 = 0u;
  v7 = (volatile signed __int32 *)*((_QWORD *)&v82 + 1);
  v82 = __PAIR128__(v5, v6);
  if ( v7 )
  {
    sub_A191D0(v7);
    if ( v83.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
    v6 = v82;
  }
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 21;
  sub_1525B40(v6, &v83);
  v83.m128i_i64[0] = 1;
  v83.m128i_i8[8] = 2;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = 0;
  v83.m128i_i8[8] = 6;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = v2;
  v83.m128i_i8[8] = v83.m128i_i8[8] & 0xF0 | 2;
  sub_1525B40(v82, &v83);
  v8 = *a1;
  v9 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v9;
  v78 = sub_15271D0(v8, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  sub_1531130(&v83);
  v10 = v83.m128i_i64[1];
  v11 = v83.m128i_i64[0];
  v83 = 0u;
  v12 = (volatile signed __int32 *)*((_QWORD *)&v82 + 1);
  v82 = __PAIR128__(v10, v11);
  if ( v12 )
  {
    sub_A191D0(v12);
    if ( v83.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
    v11 = v82;
  }
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 18;
  sub_1525B40(v11, &v83);
  v83.m128i_i64[0] = 1;
  v83.m128i_i8[8] = 2;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = 0;
  v83.m128i_i8[8] = 6;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = v2;
  v83.m128i_i8[8] = v83.m128i_i8[8] & 0xF0 | 2;
  sub_1525B40(v82, &v83);
  v13 = *a1;
  v14 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v14;
  v76 = sub_15271D0(v13, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  sub_1531130(&v83);
  v15 = v83.m128i_i64[1];
  v16 = v83.m128i_i64[0];
  v83 = 0u;
  v17 = (volatile signed __int32 *)*((_QWORD *)&v82 + 1);
  v82 = __PAIR128__(v15, v16);
  if ( v17 )
  {
    sub_A191D0(v17);
    if ( v83.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
    v16 = v82;
  }
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 19;
  sub_1525B40(v16, &v83);
  v83.m128i_i64[0] = 0;
  v83.m128i_i8[8] = 6;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = 0;
  v83.m128i_i8[8] = 8;
  sub_1525B40(v82, &v83);
  v18 = *a1;
  v19 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v19;
  v74 = sub_15271D0(v18, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  sub_1531130(&v83);
  v20 = v83.m128i_i64[1];
  v21 = v83.m128i_i64[0];
  v83 = 0u;
  v22 = (volatile signed __int32 *)*((_QWORD *)&v82 + 1);
  v82 = __PAIR128__(v20, v21);
  if ( v22 )
  {
    sub_A191D0(v22);
    if ( v83.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
    v21 = v82;
  }
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 20;
  sub_1525B40(v21, &v83);
  v83.m128i_i64[0] = 1;
  v83.m128i_i8[8] = 2;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = 0;
  v83.m128i_i8[8] = 6;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = v2;
  v83.m128i_i8[8] = v83.m128i_i8[8] & 0xF0 | 2;
  sub_1525B40(v82, &v83);
  v23 = *a1;
  v24 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v24;
  v75 = sub_15271D0(v23, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  sub_1531130(&v83);
  v25 = v83.m128i_i64[1];
  v26 = v83.m128i_i64[0];
  v83 = 0u;
  v27 = (volatile signed __int32 *)*((_QWORD *)&v82 + 1);
  v82 = __PAIR128__(v25, v26);
  if ( v27 )
  {
    sub_A191D0(v27);
    if ( v83.m128i_i64[1] )
      sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
    v26 = v82;
  }
  v83.m128i_i8[8] |= 1u;
  v83.m128i_i64[0] = 11;
  sub_1525B40(v26, &v83);
  v83.m128i_i64[0] = 8;
  v83.m128i_i8[8] = 4;
  sub_1525B40(v82, &v83);
  v83.m128i_i64[0] = v2;
  v83.m128i_i8[8] = v83.m128i_i8[8] & 0xF0 | 2;
  sub_1525B40(v82, &v83);
  v28 = *a1;
  v29 = *((_QWORD *)&v82 + 1);
  v83.m128i_i64[0] = v82;
  v82 = 0u;
  v83.m128i_i64[1] = v29;
  v77 = sub_15271D0(v28, v83.m128i_i64);
  if ( v83.m128i_i64[1] )
    sub_A191D0((volatile signed __int32 *)v83.m128i_i64[1]);
  v83.m128i_i64[0] = a1[11] - a1[10];
  sub_1525CA0((__int64)&v84, &v83);
  sub_152F3D0(*a1, 1u, (__int64)&v84, 0);
  v30 = a1[10];
  v31 = a1[11];
  LODWORD(v85) = 0;
  v32 = v31 - v30;
  if ( (_DWORD)v32 )
  {
    v33 = 0;
    v81 = 8LL * (unsigned int)(v32 - 1);
LABEL_35:
    v34 = *(__int64 *)((char *)v30 + v33);
    switch ( *(_BYTE *)(v34 + 8) )
    {
      case 0:
        v35 = 0;
        v36 = 2;
        goto LABEL_42;
      case 1:
        v35 = 0;
        v36 = 10;
        goto LABEL_42;
      case 2:
        v35 = 0;
        v36 = 3;
        goto LABEL_42;
      case 3:
        v35 = 0;
        v36 = 4;
        goto LABEL_42;
      case 4:
        v35 = 0;
        v36 = 13;
        goto LABEL_42;
      case 5:
        v35 = 0;
        v36 = 14;
        goto LABEL_42;
      case 6:
        v35 = 0;
        v36 = 15;
        goto LABEL_42;
      case 7:
        v35 = 0;
        v36 = 5;
        goto LABEL_42;
      case 8:
        v35 = 0;
        v36 = 16;
        goto LABEL_42;
      case 9:
        v35 = 0;
        v36 = 17;
        goto LABEL_42;
      case 0xA:
        v35 = 0;
        v36 = 22;
        goto LABEL_42;
      case 0xB:
        v36 = 7;
        v83.m128i_i64[0] = *(_DWORD *)(v34 + 8) >> 8;
        sub_1525CA0((__int64)&v84, &v83);
        v35 = 0;
        goto LABEL_42;
      case 0xC:
        v83.m128i_i64[0] = *(_DWORD *)(v34 + 8) >> 8 != 0;
        sub_1525CA0((__int64)&v84, &v83);
        v83.m128i_i64[0] = (unsigned int)sub_1524C80(v80, **(_QWORD **)(v34 + 16));
        sub_1525CA0((__int64)&v84, &v83);
        v48 = *(_DWORD *)(v34 + 12);
        if ( v48 == 1 )
          goto LABEL_80;
        v49 = v33;
        v50 = (unsigned int)v85;
        v51 = 8;
        v52 = 8LL * (unsigned int)(v48 - 2) + 16;
        while ( 1 )
        {
          v58 = *((unsigned int *)a1 + 18);
          v59 = a1[7];
          if ( (_DWORD)v58 )
          {
            v53 = *(__int64 **)(*(_QWORD *)(v34 + 16) + v51);
            v54 = (v58 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
            v55 = &v59[2 * v54];
            v56 = *v55;
            if ( v53 == *v55 )
            {
LABEL_68:
              v57 = (unsigned int)(*((_DWORD *)v55 + 2) - 1);
              if ( HIDWORD(v85) <= (unsigned int)v50 )
                goto LABEL_72;
              goto LABEL_69;
            }
            v61 = 1;
            while ( v56 != (__int64 *)-8LL )
            {
              v54 = (v58 - 1) & (v61 + v54);
              v72 = v61 + 1;
              v55 = &v59[2 * v54];
              v56 = *v55;
              if ( v53 == *v55 )
                goto LABEL_68;
              v61 = v72;
            }
          }
          v57 = (unsigned int)(LODWORD(v59[2 * v58 + 1]) - 1);
          if ( HIDWORD(v85) <= (unsigned int)v50 )
          {
LABEL_72:
            v68 = v49;
            v70 = v57;
            sub_16CD150(&v84, v86, 0, 8);
            v50 = (unsigned int)v85;
            v49 = v68;
            v57 = v70;
          }
LABEL_69:
          v51 += 8;
          *(_QWORD *)&v84[8 * v50] = v57;
          v50 = (unsigned int)(v85 + 1);
          LODWORD(v85) = v85 + 1;
          if ( v52 == v51 )
          {
            v33 = v49;
LABEL_80:
            v35 = v78;
            v36 = 21;
            goto LABEL_42;
          }
        }
      case 0xD:
        v83.m128i_i64[0] = (*(_DWORD *)(v34 + 8) >> 9) & 1;
        sub_1525CA0((__int64)&v84, &v83);
        v38 = *(__int64 ***)(v34 + 16);
        v39 = (unsigned int)v85;
        if ( &v38[*(unsigned int *)(v34 + 12)] == v38 )
          goto LABEL_77;
        v40 = v34;
        v41 = &v38[*(unsigned int *)(v34 + 12)];
        break;
      case 0xE:
        v83.m128i_i64[0] = *(_QWORD *)(v34 + 32);
        sub_1525CA0((__int64)&v84, &v83);
        v36 = 11;
        v83.m128i_i64[0] = (unsigned int)sub_1524C80(v80, *(_QWORD *)(v34 + 24));
        sub_1525CA0((__int64)&v84, &v83);
        v35 = v77;
        goto LABEL_42;
      case 0xF:
        v83.m128i_i64[0] = (unsigned int)sub_1524C80(v80, *(_QWORD *)(v34 + 24));
        sub_1525CA0((__int64)&v84, &v83);
        v36 = 8;
        v37 = *(_DWORD *)(v34 + 8) >> 8;
        v83.m128i_i64[0] = v37;
        sub_1525CA0((__int64)&v84, &v83);
        v35 = 0;
        if ( !v37 )
          v35 = v79;
        goto LABEL_42;
      case 0x10:
        v83.m128i_i64[0] = *(_QWORD *)(v34 + 32);
        sub_1525CA0((__int64)&v84, &v83);
        v36 = 12;
        v83.m128i_i64[0] = (unsigned int)sub_1524C80(v80, *(_QWORD *)(v34 + 24));
        sub_1525CA0((__int64)&v84, &v83);
        v35 = 0;
        goto LABEL_42;
      default:
        v35 = 0;
        v36 = 0;
        goto LABEL_42;
    }
    while ( 1 )
    {
      v46 = *((unsigned int *)a1 + 18);
      v47 = a1[7];
      if ( (_DWORD)v46 )
      {
        v42 = (v46 - 1) & (((unsigned int)*v38 >> 9) ^ ((unsigned int)*v38 >> 4));
        v43 = &v47[2 * v42];
        v44 = *v43;
        if ( *v38 == *v43 )
        {
LABEL_52:
          v45 = (unsigned int)(*((_DWORD *)v43 + 2) - 1);
          if ( HIDWORD(v85) <= (unsigned int)v39 )
            goto LABEL_56;
          goto LABEL_53;
        }
        v60 = 1;
        while ( v44 != (__int64 *)-8LL )
        {
          v42 = (v46 - 1) & (v60 + v42);
          v71 = v60 + 1;
          v43 = &v47[2 * v42];
          v44 = *v43;
          if ( *v38 == *v43 )
            goto LABEL_52;
          v60 = v71;
        }
      }
      v45 = (unsigned int)(LODWORD(v47[2 * v46 + 1]) - 1);
      if ( HIDWORD(v85) <= (unsigned int)v39 )
      {
LABEL_56:
        v67 = v40;
        v69 = v45;
        sub_16CD150(&v84, v86, 0, 8);
        v39 = (unsigned int)v85;
        v40 = v67;
        v45 = v69;
      }
LABEL_53:
      ++v38;
      *(_QWORD *)&v84[8 * v39] = v45;
      v39 = (unsigned int)(v85 + 1);
      LODWORD(v85) = v85 + 1;
      if ( v38 == v41 )
      {
        v34 = v40;
LABEL_77:
        if ( (*(_DWORD *)(v34 + 8) & 0x400) != 0 )
        {
          v35 = v76;
          v36 = 18;
        }
        else
        {
          v62 = 0;
          if ( (*(_DWORD *)(v34 + 8) & 0x100) != 0 )
            v62 = v75;
          v63 = -((*(_DWORD *)(v34 + 8) & 0x100) != 0);
          v73 = v62;
          sub_1643640(v34);
          v35 = v73;
          v36 = (v63 & 0xE) + 6;
          if ( v64 )
          {
            v65 = (char *)sub_1643640(v34);
            sub_1528330(*a1, 0x13u, v65, v66, v74);
            v35 = v73;
          }
        }
LABEL_42:
        sub_152F3D0(*a1, v36, (__int64)&v84, v35);
        LODWORD(v85) = 0;
        if ( v33 == v81 )
          break;
        v30 = a1[10];
        v33 += 8;
        goto LABEL_35;
      }
    }
  }
  sub_15263C0(*a1);
  if ( *((_QWORD *)&v82 + 1) )
    sub_A191D0(*((volatile signed __int32 **)&v82 + 1));
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
}
