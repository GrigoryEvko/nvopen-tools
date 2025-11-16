// Function: sub_17D6E80
// Address: 0x17d6e80
//
_QWORD *__fastcall sub_17D6E80(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  _BYTE *v6; // r13
  __int64 **v7; // r15
  char v8; // al
  int v9; // ecx
  __int64 v10; // rax
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // r12
  char v15; // al
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rtt
  unsigned __int64 v18; // rax
  int v19; // ebx
  int v20; // eax
  __int64 *v21; // r14
  unsigned __int64 v22; // r12
  _QWORD *v23; // rdi
  unsigned __int64 v24; // r8
  char v25; // al
  unsigned __int64 v26; // r12
  __int64 v27; // r12
  unsigned int v28; // eax
  unsigned __int64 v29; // r11
  unsigned __int64 v30; // r11
  __int64 *v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 *v36; // rax
  __int64 v37; // rax
  __int128 v38; // rdi
  __int64 *v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 *v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rbx
  __int64 *v47; // rax
  _QWORD *v48; // r14
  __int64 v49; // rax
  __int64 *v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *result; // rax
  __int64 v55; // [rsp+8h] [rbp-D8h]
  int v56; // [rsp+8h] [rbp-D8h]
  __int64 v57; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v58; // [rsp+10h] [rbp-D0h]
  __int64 v59; // [rsp+10h] [rbp-D0h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  __int64 v61; // [rsp+10h] [rbp-D0h]
  __int64 v62; // [rsp+10h] [rbp-D0h]
  __int64 v63; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v64; // [rsp+18h] [rbp-C8h]
  __int64 v67; // [rsp+38h] [rbp-A8h]
  int v68; // [rsp+40h] [rbp-A0h]
  unsigned int v69; // [rsp+44h] [rbp-9Ch]
  unsigned int v70; // [rsp+48h] [rbp-98h]
  _QWORD v71[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v72; // [rsp+60h] [rbp-80h]
  _QWORD *v73; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v74[2]; // [rsp+80h] [rbp-60h] BYREF
  int v75; // [rsp+90h] [rbp-50h]

  v4 = 48;
  v5 = *(_QWORD *)(a1[1] + 40LL);
  v72 = 260;
  v71[0] = v5 + 240;
  sub_16E1010((__int64)&v73, (__int64)v71);
  if ( v75 != 17 )
    v4 = 32;
  v6 = (_BYTE *)sub_1632FA0(*(_QWORD *)(a1[1] + 40LL));
  v7 = (__int64 **)((*a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  v64 = sub_1389B50(a2);
  if ( (__int64 **)v64 != v7 )
  {
    v68 = v4;
    v67 = a3;
    while ( 1 )
    {
      v21 = *v7;
      v22 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = (_QWORD *)(v22 + 56);
      v24 = v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
      v70 = -1431655765 * ((__int64)((__int64)v7 - v24) >> 3);
      v69 = *(_DWORD *)(*(_QWORD *)(v22 + 64) + 12LL) - 1;
      v58 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v7 - v24) >> 3);
      if ( (*a2 & 4) != 0 )
        break;
      v25 = sub_1560290(v23, v70, 6);
      v9 = v58;
      if ( v25 )
      {
LABEL_23:
        v59 = **(_QWORD **)(*v21 + 16);
        v26 = (unsigned int)sub_15A9FE0((__int64)v6, v59);
        v27 = (v26 + ((unsigned __int64)(sub_127FA20((__int64)v6, v59) + 7) >> 3) - 1) / v26 * v26;
        v28 = sub_15603A0((_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56), v70);
        v29 = v28;
        if ( v28 <= 7 )
          v29 = 8;
        v30 = (v29 + v4 - 1) / v29 * v29;
        if ( v70 >= v69 )
        {
          v72 = 257;
          v56 = v30;
          v41 = sub_12A95D0((__int64 *)v67, *(_QWORD *)(a1[2] + 224LL), *(_QWORD *)(a1[2] + 176LL), (__int64)v71);
          v72 = 257;
          v42 = sub_15A0680(*(_QWORD *)(a1[2] + 176LL), v56 - v68, 0);
          v43 = sub_12899C0((__int64 *)v67, v41, v42, (__int64)v71, 0, 0);
          v71[0] = "_msarg";
          v72 = 259;
          v44 = sub_17CD8D0((_QWORD *)a1[3], v59);
          v45 = sub_1646BA0(v44, 0);
          v46 = (_QWORD *)sub_12AA3B0((__int64 *)v67, 0x2Eu, v43, v45, (__int64)v71);
          v63 = a1[3];
          v47 = (__int64 *)sub_1643330(*(_QWORD **)(v67 + 24));
          v48 = (_QWORD *)sub_17CFB40(v63, (__int64)v21, (__int64 *)v67, v47, 8u);
          v49 = sub_1643360(*(_QWORD **)(v67 + 24));
          v50 = (__int64 *)sub_159C470(v49, v27, 0);
          sub_15E7430((__int64 *)v67, v46, 8u, v48, 8u, v50, 0, 0, 0, 0, 0);
          LODWORD(v30) = v56;
        }
        v4 = (unsigned int)v30 + (((_DWORD)v27 + 7) & 0xFFFFFFF8);
        goto LABEL_16;
      }
      v10 = *(_QWORD *)(v22 - 72);
      if ( !*(_BYTE *)(v10 + 16) )
        goto LABEL_7;
LABEL_8:
      v57 = *v21;
      v11 = (unsigned int)sub_15A9FE0((__int64)v6, *v21);
      v12 = sub_127FA20((__int64)v6, v57);
      v13 = *v21;
      v14 = (v11 + ((unsigned __int64)(v12 + 7) >> 3) - 1) / v11 * v11;
      v15 = *(_BYTE *)(*v21 + 8);
      if ( v15 == 14 )
      {
        v31 = *(__int64 **)(v13 + 16);
        v16 = 8;
        v13 = *v31;
        if ( *(_BYTE *)(*v31 + 8) != 6 )
        {
LABEL_29:
          v32 = sub_12BE0A0((__int64)v6, v13);
          v16 = 8;
          if ( v32 >= 8 )
            v16 = v32;
        }
      }
      else
      {
        v16 = 8;
        if ( v15 == 16 )
          goto LABEL_29;
      }
      v17 = v16 + v4 - 1;
      v18 = v16 * (v17 / v16);
      v19 = v16 * (v17 / v16);
      if ( v14 <= 7 && *v6 )
        v19 = v18 - v14 + 8;
      if ( v70 >= v69 )
      {
        v33 = *v21;
        v72 = 257;
        v55 = v33;
        v60 = sub_12A95D0((__int64 *)v67, *(_QWORD *)(a1[2] + 224LL), *(_QWORD *)(a1[2] + 176LL), (__int64)v71);
        v72 = 257;
        v34 = sub_15A0680(*(_QWORD *)(a1[2] + 176LL), v19 - v68, 0);
        v61 = sub_12899C0((__int64 *)v67, v60, v34, (__int64)v71, 0, 0);
        v71[0] = "_msarg";
        v35 = (_QWORD *)a1[3];
        v72 = 259;
        v36 = sub_17CD8D0(v35, v55);
        v37 = sub_1646BA0(v36, 0);
        *((_QWORD *)&v38 + 1) = v21;
        v62 = sub_12AA3B0((__int64 *)v67, 0x2Eu, v61, v37, (__int64)v71);
        *(_QWORD *)&v38 = a1[3];
        v39 = sub_17D4DA0(v38);
        v40 = sub_12A8F50((__int64 *)v67, (__int64)v39, v62, 0);
        sub_15F9450((__int64)v40, 8u);
      }
      v4 = ((_DWORD)v14 + v19 + 7) & 0xFFFFFFF8;
LABEL_16:
      v20 = v68;
      if ( v70 < v69 )
        v20 = v4;
      v7 += 3;
      v68 = v20;
      if ( (__int64 **)v64 == v7 )
      {
        a3 = v67;
        v51 = (unsigned int)(v4 - v20);
        goto LABEL_33;
      }
    }
    v8 = sub_1560290(v23, v70, 6);
    v9 = v58;
    if ( v8 )
      goto LABEL_23;
    v10 = *(_QWORD *)(v22 - 24);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_8;
LABEL_7:
    v71[0] = *(_QWORD *)(v10 + 112);
    if ( (unsigned __int8)sub_1560290(v71, v9, 6) )
      goto LABEL_23;
    goto LABEL_8;
  }
  v51 = 0;
LABEL_33:
  v52 = sub_1643360(*(_QWORD **)(a3 + 24));
  v53 = sub_159C470(v52, v51, 0);
  sub_12A8F50((__int64 *)a3, v53, *(_QWORD *)(a1[2] + 232LL), 0);
  result = v74;
  if ( v73 != v74 )
    return (_QWORD *)j_j___libc_free_0(v73, v74[0] + 1LL);
  return result;
}
