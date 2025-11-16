// Function: sub_1471910
// Address: 0x1471910
//
__int64 __fastcall sub_1471910(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbx
  __int16 v4; // ax
  _QWORD *v6; // r13
  __int64 v7; // rax
  unsigned __int8 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  _QWORD *v11; // r14
  __int64 **v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r10
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v23; // rax
  _BYTE *v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rax
  _BYTE *v27; // r14
  _QWORD *v28; // r14
  unsigned int v29; // r13d
  __int64 v30; // rdx
  __int16 v31; // ax
  _BYTE *v32; // rdi
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rcx
  char v36; // al
  __int64 v37; // rsi
  __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // r8
  unsigned int v43; // r14d
  __int64 v44; // rdx
  _QWORD *v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // r14
  bool v49; // al
  __int64 v50; // rcx
  unsigned int v51; // eax
  _QWORD *v52; // rcx
  __int64 v53; // r13
  _QWORD *v54; // r14
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rsi
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rsi
  __int64 v62; // [rsp+0h] [rbp-D0h]
  __int64 v63; // [rsp+10h] [rbp-C0h]
  __int64 v64; // [rsp+20h] [rbp-B0h]
  _QWORD *v66; // [rsp+28h] [rbp-A8h]
  int v67; // [rsp+28h] [rbp-A8h]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  char v69; // [rsp+30h] [rbp-A0h]
  int v70; // [rsp+30h] [rbp-A0h]
  __int64 v71; // [rsp+30h] [rbp-A0h]
  __int64 v72; // [rsp+30h] [rbp-A0h]
  __int64 v73; // [rsp+38h] [rbp-98h]
  __int64 v74; // [rsp+38h] [rbp-98h]
  __int64 v75; // [rsp+38h] [rbp-98h]
  __int64 v76; // [rsp+38h] [rbp-98h]
  __int64 v77; // [rsp+40h] [rbp-90h] BYREF
  __int64 v78; // [rsp+48h] [rbp-88h] BYREF
  _QWORD *v79; // [rsp+50h] [rbp-80h] BYREF
  __int64 v80; // [rsp+58h] [rbp-78h]
  _BYTE v81[112]; // [rsp+60h] [rbp-70h] BYREF

  v3 = a2;
  v4 = *(_WORD *)(a2 + 24);
  if ( !v4 )
    return v3;
  v6 = a3;
  if ( v4 != 10 )
  {
    if ( ((v4 - 4) & 0xFFFA) == 0 )
    {
      v23 = *(_QWORD *)(a2 + 40);
      v67 = v23;
      v70 = v23;
      if ( !(_DWORD)v23 )
        return v3;
      v24 = *(_BYTE **)(a2 + 32);
      v25 = 0;
      v75 = (unsigned int)v23;
      while ( 1 )
      {
        v26 = sub_1472270(a1, *(_QWORD *)&v24[8 * v25], v6);
        v24 = *(_BYTE **)(v3 + 32);
        v78 = v26;
        v27 = &v24[8 * v25];
        if ( v26 != *(_QWORD *)v27 )
          break;
        if ( v75 == ++v25 )
          return v3;
      }
      v79 = v81;
      v80 = 0x800000000LL;
      sub_145C5B0((__int64)&v79, v24, v27);
      sub_1458920((__int64)&v79, &v78);
      if ( v67 != (_DWORD)v25 + 1 )
      {
        v28 = v6;
        v29 = v25 + 1;
        do
        {
          v30 = v29++;
          v78 = sub_1472270(a1, *(_QWORD *)(*(_QWORD *)(v3 + 32) + 8 * v30), v28);
          sub_1458920((__int64)&v79, &v78);
        }
        while ( v70 != v29 );
      }
      v31 = *(_WORD *)(v3 + 24);
      switch ( v31 )
      {
        case 4:
          v21 = sub_147DD40(a1, &v79, 0, 0);
          break;
        case 5:
          v21 = sub_147EE30(a1, &v79, 0, 0);
          break;
        case 9:
          v21 = sub_147A3C0(a1, &v79);
          break;
        default:
          v21 = sub_14813B0(a1, &v79);
          break;
      }
LABEL_39:
      v32 = v79;
      goto LABEL_40;
    }
    if ( v4 == 6 )
    {
      v33 = sub_1472270(a1, *(_QWORD *)(a2 + 32), a3);
      v34 = sub_1472270(a1, *(_QWORD *)(a2 + 40), v6);
      if ( v33 != *(_QWORD *)(a2 + 32) || v34 != *(_QWORD *)(a2 + 40) )
        return sub_1483CF0(a1, v33, v34);
      return v3;
    }
    if ( v4 != 7 )
    {
      v57 = *(_QWORD *)(a2 + 32);
      if ( v4 == 2 )
      {
        v58 = sub_1472270(a1, v57, a3);
        if ( v58 != *(_QWORD *)(v3 + 32) )
          return sub_14747F0(a1, v58, *(_QWORD *)(v3 + 40), 0);
      }
      else if ( v4 == 3 )
      {
        v59 = sub_1472270(a1, v57, a3);
        if ( v59 != *(_QWORD *)(v3 + 32) )
          return sub_147B0D0(a1, v59, *(_QWORD *)(v3 + 40), 0);
      }
      else
      {
        v60 = sub_1472270(a1, v57, a3);
        if ( v60 != *(_QWORD *)(v3 + 32) )
          return sub_14835F0(a1, v60, *(_QWORD *)(v3 + 40), 0);
      }
      return v3;
    }
    v71 = *(_QWORD *)(a2 + 40);
    if ( (_DWORD)v71 )
    {
      v39 = *(_BYTE **)(a2 + 32);
      v40 = 0;
      while ( 1 )
      {
        v41 = sub_1472270(a1, *(_QWORD *)&v39[8 * v40], v6);
        v39 = *(_BYTE **)(v3 + 32);
        v77 = v41;
        if ( v41 != *(_QWORD *)&v39[8 * v40] )
          break;
        if ( ++v40 == (unsigned int)v71 )
          goto LABEL_91;
      }
      v79 = v81;
      v80 = 0x800000000LL;
      sub_145C5B0((__int64)&v79, v39, &v39[8 * v40]);
      sub_1458920((__int64)&v79, &v77);
      if ( (_DWORD)v71 != (_DWORD)v40 + 1 )
      {
        v43 = v40 + 1;
        do
        {
          v44 = v43++;
          v78 = sub_1472270(a1, *(_QWORD *)(*(_QWORD *)(v3 + 32) + 8 * v44), v6);
          sub_1458920((__int64)&v79, &v78);
        }
        while ( (_DWORD)v71 != v43 );
      }
      v21 = sub_14785F0(a1, &v79, *(_QWORD *)(v3 + 48), *(_WORD *)(v3 + 26) & 1, v42);
      if ( *(_WORD *)(v21 + 24) != 7 )
        goto LABEL_39;
      if ( v79 != (_QWORD *)v81 )
        _libc_free((unsigned __int64)v79);
    }
    else
    {
LABEL_91:
      v21 = v3;
    }
    v45 = *(_QWORD **)(v21 + 48);
    if ( v45 != v6 )
    {
      while ( v6 )
      {
        v6 = (_QWORD *)*v6;
        if ( v45 == v6 )
          return v21;
      }
      v46 = sub_1481F60(a1, v45);
      if ( v46 != sub_1456E90((__int64)a1) )
        return sub_1487810(v21, v46, a1);
    }
    return v21;
  }
  v7 = *(_QWORD *)(a2 - 8);
  v8 = *(_BYTE *)(v7 + 16);
  v63 = v7;
  if ( v8 <= 0x17u )
    return v3;
  v73 = *(_QWORD *)(v7 + 40);
  v9 = sub_13AE450(a1[8], v73);
  if ( !v9 )
    goto LABEL_7;
  if ( *(_QWORD **)v9 != v6 )
    goto LABEL_7;
  if ( v8 != 77 )
    goto LABEL_7;
  if ( v73 != **(_QWORD **)(v9 + 32) )
    goto LABEL_7;
  v72 = v9;
  v47 = sub_1481F60(a1, v9);
  v76 = v47;
  if ( *(_WORD *)(v47 + 24) )
    goto LABEL_7;
  v48 = *(_QWORD *)(v47 + 32) + 24LL;
  v49 = sub_13D01C0(v48);
  v50 = v72;
  if ( !v49 || (v51 = *(_DWORD *)(v63 + 20) & 0xFFFFFFF) == 0 )
  {
LABEL_98:
    v61 = sub_146B5E0((__int64)a1, v63, v48, v50);
    if ( v61 )
      return sub_146F1B0((__int64)a1, v61);
LABEL_7:
    if ( !(unsigned __int8)sub_1452C00(v63) )
      return v3;
    v79 = v81;
    v80 = 0x400000000LL;
    v10 = sub_13CF970(v63);
    v74 = v10 + 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF);
    if ( v10 == v74 )
      return v3;
    v62 = v3;
    v11 = v6;
    v12 = (__int64 **)v10;
    v69 = 0;
    do
    {
      v14 = *v12;
      if ( *((_BYTE *)*v12 + 16) <= 0x10u )
      {
        v13 = (unsigned int)v80;
        if ( (unsigned int)v80 >= HIDWORD(v80) )
        {
          sub_16CD150(&v79, v81, 0, 8);
          v13 = (unsigned int)v80;
        }
        v79[v13] = v14;
        LODWORD(v80) = v80 + 1;
      }
      else
      {
        if ( !sub_1456C80((__int64)a1, *v14)
          || (v15 = sub_146F1B0((__int64)a1, (__int64)v14),
              v16 = sub_1472270(a1, v15, v11),
              v69 |= v15 != v16,
              v17 = (_QWORD *)sub_14526C0(v16),
              (v18 = (__int64)v17) == 0) )
        {
          v32 = v79;
          v21 = v62;
          goto LABEL_40;
        }
        if ( *v17 != *v14 )
        {
          v64 = *v14;
          v66 = v17;
          v19 = sub_15FBEB0(v17, 0, *v14, 0);
          v18 = sub_15A46C0(v19, v66, v64, 0);
        }
        v20 = (unsigned int)v80;
        if ( (unsigned int)v80 >= HIDWORD(v80) )
        {
          v68 = v18;
          sub_16CD150(&v79, v81, 0, 8);
          v20 = (unsigned int)v80;
          v18 = v68;
        }
        v79[v20] = v18;
        LODWORD(v80) = v80 + 1;
      }
      v12 += 3;
    }
    while ( (__int64 **)v74 != v12 );
    v3 = v62;
    if ( !v69 )
    {
      if ( v79 != (_QWORD *)v81 )
        _libc_free((unsigned __int64)v79);
      return v3;
    }
    v35 = sub_1632FA0(*(_QWORD *)(a1[3] + 40LL));
    v36 = *(_BYTE *)(v63 + 16);
    if ( (unsigned __int8)(v36 - 75) > 1u )
    {
      v32 = v79;
      if ( v36 == 54 )
      {
        if ( (*(_BYTE *)(v63 + 18) & 1) != 0 )
          goto LABEL_83;
        v37 = sub_14D8290(*v79, *(_QWORD *)v63, v35);
      }
      else
      {
        v37 = sub_14DD1F0(v63, v79, (unsigned int)v80, v35, a1[5]);
      }
    }
    else
    {
      v37 = sub_14D7760(*(_WORD *)(v63 + 18) & 0x7FFF, *v79, v79[1], v35, a1[5]);
    }
    v32 = v79;
    if ( v37 )
    {
      v38 = sub_146F1B0((__int64)a1, v37);
      v32 = v79;
      v21 = v38;
      goto LABEL_40;
    }
LABEL_83:
    v21 = v62;
LABEL_40:
    if ( v32 != v81 )
      _libc_free((unsigned __int64)v32);
    return v21;
  }
  v52 = v6;
  v53 = 0;
  v54 = v52;
  while ( 1 )
  {
    v55 = (*(_BYTE *)(v63 + 23) & 0x40) != 0 ? *(_QWORD *)(v63 - 8) : v63 - 24LL * v51;
    if ( !sub_1377F70(v72 + 56, *(_QWORD *)(v55 + 8 * v53 + 24LL * *(unsigned int *)(v63 + 56) + 8)) )
    {
      v56 = sub_1455F60(v63, v53);
      if ( v56 )
        return sub_146F1B0((__int64)a1, v56);
    }
    ++v53;
    v51 = *(_DWORD *)(v63 + 20) & 0xFFFFFFF;
    if ( v51 <= (unsigned int)v53 )
    {
      v6 = v54;
      v50 = v72;
      v3 = a2;
      v48 = *(_QWORD *)(v76 + 32) + 24LL;
      goto LABEL_98;
    }
  }
}
