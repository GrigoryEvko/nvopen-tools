// Function: sub_1B4DD20
// Address: 0x1b4dd20
//
_QWORD *__fastcall sub_1B4DD20(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v6; // r15
  unsigned int v9; // eax
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r13
  unsigned int v14; // r14d
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r13
  __int64 **v23; // rsi
  __int64 v24; // rax
  unsigned int v25; // r14d
  bool v26; // al
  __int64 v27; // r13
  unsigned int v28; // ebx
  bool v29; // al
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r11
  _QWORD *v35; // r13
  __int64 v36; // rdi
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rcx
  _QWORD *v40; // rax
  __int64 v41; // r11
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // r11
  __int64 *v46; // r10
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned __int64 *v49; // r15
  __int64 v50; // rax
  unsigned __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 *v56; // r13
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdi
  unsigned __int64 *v62; // r15
  __int64 v63; // rax
  unsigned __int64 v64; // rcx
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // rdx
  unsigned __int8 *v68; // rsi
  int v69; // [rsp+4h] [rbp-9Ch]
  __int64 v70; // [rsp+8h] [rbp-98h]
  __int64 v71; // [rsp+8h] [rbp-98h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  __int64 v73; // [rsp+18h] [rbp-88h]
  __int64 v74; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v75; // [rsp+20h] [rbp-80h] BYREF
  __int64 v76; // [rsp+28h] [rbp-78h]
  __int64 v77[2]; // [rsp+30h] [rbp-70h] BYREF
  char v78; // [rsp+40h] [rbp-60h]
  char v79; // [rsp+41h] [rbp-5Fh]
  __int64 v80[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v81; // [rsp+60h] [rbp-40h]

  v6 = a2;
  v9 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 == 2 )
  {
    v11 = *(__int64 **)(a1 + 16);
    v12 = *(_QWORD *)a2;
    v13 = *v11;
    v80[0] = (__int64)"switch.cast";
    v81 = 259;
    v14 = sub_16431D0(v12);
    v15 = sub_16431D0(v13);
    if ( v14 < v15 )
    {
      v6 = sub_12AA3B0((__int64 *)a3, 0x25u, a2, v13, (__int64)v80);
    }
    else if ( v14 > v15 )
    {
      v6 = sub_12AA3B0((__int64 *)a3, 0x24u, a2, v13, (__int64)v80);
    }
    v79 = 1;
    v77[0] = (__int64)"switch.shiftamt";
    v16 = *(_QWORD *)(a1 + 24);
    v78 = 3;
    v17 = sub_159C470(v13, *(_DWORD *)(v16 + 8) >> 8, 0);
    if ( *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(v17 + 16) > 0x10u )
    {
      v81 = 257;
      v60 = sub_15FB440(15, (__int64 *)v6, v17, (__int64)v80, 0);
      v61 = *(_QWORD *)(a3 + 8);
      v18 = (_QWORD *)v60;
      if ( v61 )
      {
        v62 = *(unsigned __int64 **)(a3 + 16);
        sub_157E9D0(v61 + 40, v60);
        v63 = v18[3];
        v64 = *v62;
        v18[4] = v62;
        v64 &= 0xFFFFFFFFFFFFFFF8LL;
        v18[3] = v64 | v63 & 7;
        *(_QWORD *)(v64 + 8) = v18 + 3;
        *v62 = *v62 & 7 | (unsigned __int64)(v18 + 3);
      }
      sub_164B780((__int64)v18, v77);
      v65 = *(_QWORD *)a3;
      if ( *(_QWORD *)a3 )
      {
        v75 = *(unsigned __int8 **)a3;
        sub_1623A60((__int64)&v75, v65, 2);
        v66 = v18[6];
        v67 = (__int64)(v18 + 6);
        if ( v66 )
        {
          sub_161E7C0((__int64)(v18 + 6), v66);
          v67 = (__int64)(v18 + 6);
        }
        v68 = v75;
        v18[6] = v75;
        if ( v68 )
          sub_1623210((__int64)&v75, v68, v67);
      }
    }
    else
    {
      v18 = (_QWORD *)sub_15A2C20((__int64 *)v6, v17, 0, 0, a4, a5, a6);
    }
    v19 = *(_QWORD *)(a1 + 16);
    v80[0] = (__int64)"switch.downshift";
    v81 = 259;
    v20 = sub_156E320((__int64 *)a3, v19, (__int64)v18, (__int64)v80, 0);
    v21 = *(_QWORD *)(a1 + 24);
    v80[0] = (__int64)"switch.masked";
    v81 = 259;
    return (_QWORD *)sub_12AA3B0((__int64 *)a3, 0x24u, v20, v21, (__int64)v80);
  }
  else if ( v9 > 2 )
  {
    v30 = *(_QWORD *)a2;
    if ( (unsigned __int64)(1LL << (BYTE1(*(_DWORD *)(*(_QWORD *)a2 + 8LL)) - 1)) < *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 48) - 24LL)
                                                                                              + 32LL) )
    {
      v80[0] = (__int64)"switch.tableidx.zext";
      v81 = 259;
      v52 = sub_1644900(*(_QWORD **)v30, (*(_DWORD *)(v30 + 8) >> 8) + 1);
      v6 = sub_12AA3B0((__int64 *)a3, 0x25u, a2, v52, (__int64)v80);
    }
    v31 = sub_1643350(*(_QWORD **)(a3 + 24));
    v32 = sub_159C470(v31, 0, 0);
    v33 = *(_QWORD *)(a1 + 48);
    v76 = v6;
    v75 = (unsigned __int8 *)v32;
    v79 = 1;
    v77[0] = (__int64)"switch.gep";
    v78 = 3;
    v34 = *(_QWORD *)(v33 + 24);
    if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v32 + 16) > 0x10u || *(_BYTE *)(v6 + 16) > 0x10u )
    {
      v81 = 257;
      if ( !v34 )
      {
        v53 = *(_QWORD *)v33;
        if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) == 16 )
          v53 = **(_QWORD **)(v53 + 16);
        v34 = *(_QWORD *)(v53 + 24);
      }
      v73 = v34;
      v40 = sub_1648A60(72, 3u);
      v41 = v73;
      v35 = v40;
      if ( v40 )
      {
        v74 = (__int64)v40;
        v72 = (__int64)(v40 - 9);
        v42 = *(_QWORD *)v33;
        if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) == 16 )
          v42 = **(_QWORD **)(v42 + 16);
        v70 = v41;
        v69 = *(_DWORD *)(v42 + 8) >> 8;
        v43 = (__int64 *)sub_15F9F50(v41, (__int64)&v75, 2);
        v44 = sub_1646BA0(v43, v69);
        v45 = v70;
        v46 = (__int64 *)v44;
        v47 = *(_QWORD *)v33;
        if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) == 16
          || (v47 = *(_QWORD *)v75, *(_BYTE *)(*(_QWORD *)v75 + 8LL) == 16)
          || (v47 = *(_QWORD *)v76, *(_BYTE *)(*(_QWORD *)v76 + 8LL) == 16) )
        {
          v59 = sub_16463B0(v46, *(_QWORD *)(v47 + 32));
          v45 = v70;
          v46 = v59;
        }
        v71 = v45;
        sub_15F1EA0((__int64)v35, (__int64)v46, 32, v72, 3, 0);
        v35[7] = v71;
        v35[8] = sub_15F9F50(v71, (__int64)&v75, 2);
        sub_15F9CE0((__int64)v35, v33, (__int64 *)&v75, 2, (__int64)v80);
      }
      else
      {
        v74 = 0;
      }
      sub_15FA2E0((__int64)v35, 1);
      v48 = *(_QWORD *)(a3 + 8);
      if ( v48 )
      {
        v49 = *(unsigned __int64 **)(a3 + 16);
        sub_157E9D0(v48 + 40, (__int64)v35);
        v50 = v35[3];
        v51 = *v49;
        v35[4] = v49;
        v51 &= 0xFFFFFFFFFFFFFFF8LL;
        v35[3] = v51 | v50 & 7;
        *(_QWORD *)(v51 + 8) = v35 + 3;
        *v49 = *v49 & 7 | (unsigned __int64)(v35 + 3);
      }
      sub_164B780(v74, v77);
      sub_12A86E0((__int64 *)a3, (__int64)v35);
    }
    else
    {
      BYTE4(v80[0]) = 0;
      v35 = (_QWORD *)sub_15A2E80(v34, v33, (__int64 **)&v75, 2u, 1u, (__int64)v80, 0);
    }
    v80[0] = (__int64)"switch.load";
    v81 = 259;
    v6 = (__int64)sub_1648A60(64, 1u);
    if ( v6 )
      sub_15F9210(v6, *(_QWORD *)(*v35 + 24LL), (__int64)v35, 0, 0, 0);
    v36 = *(_QWORD *)(a3 + 8);
    if ( v36 )
    {
      v37 = *(__int64 **)(a3 + 16);
      sub_157E9D0(v36 + 40, v6);
      v38 = *(_QWORD *)(v6 + 24);
      v39 = *v37;
      *(_QWORD *)(v6 + 32) = v37;
      v39 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v6 + 24) = v39 | v38 & 7;
      *(_QWORD *)(v39 + 8) = v6 + 24;
      *v37 = *v37 & 7 | (v6 + 24);
    }
    sub_164B780(v6, v80);
    sub_12A86E0((__int64 *)a3, v6);
  }
  else if ( v9 )
  {
    v22 = *(_QWORD *)(a1 + 40);
    v79 = 1;
    v77[0] = (__int64)"switch.idx.cast";
    v78 = 3;
    v23 = *(__int64 ***)v22;
    if ( *(_QWORD *)v22 != *(_QWORD *)v6 )
    {
      if ( *(_BYTE *)(v6 + 16) > 0x10u )
      {
        v81 = 257;
        v54 = sub_15FE0A0((_QWORD *)v6, (__int64)v23, 0, (__int64)v80, 0);
        v55 = *(_QWORD *)(a3 + 8);
        v6 = v54;
        if ( v55 )
        {
          v56 = *(__int64 **)(a3 + 16);
          sub_157E9D0(v55 + 40, v54);
          v57 = *(_QWORD *)(v6 + 24);
          v58 = *v56;
          *(_QWORD *)(v6 + 32) = v56;
          v58 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v6 + 24) = v58 | v57 & 7;
          *(_QWORD *)(v58 + 8) = v6 + 24;
          *v56 = *v56 & 7 | (v6 + 24);
        }
        sub_164B780(v6, v77);
        sub_12A86E0((__int64 *)a3, v6);
        v22 = *(_QWORD *)(a1 + 40);
      }
      else
      {
        v24 = sub_15A4750((__int64 ***)v6, v23, 0);
        v22 = *(_QWORD *)(a1 + 40);
        v6 = v24;
      }
    }
    v25 = *(_DWORD *)(v22 + 32);
    if ( v25 <= 0x40 )
      v26 = *(_QWORD *)(v22 + 24) == 1;
    else
      v26 = v25 - 1 == (unsigned int)sub_16A57B0(v22 + 24);
    if ( !v26 )
    {
      v80[0] = (__int64)"switch.idx.mult";
      v81 = 259;
      if ( *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(v22 + 16) > 0x10u )
        v6 = (__int64)sub_17D2EF0((__int64 *)a3, 15, (__int64 *)v6, v22, v80, 0, 0);
      else
        v6 = sub_15A2C20((__int64 *)v6, v22, 0, 0, a4, a5, a6);
    }
    v27 = *(_QWORD *)(a1 + 32);
    v28 = *(_DWORD *)(v27 + 32);
    if ( v28 <= 0x40 )
      v29 = *(_QWORD *)(v27 + 24) == 0;
    else
      v29 = v28 == (unsigned int)sub_16A57B0(v27 + 24);
    if ( !v29 )
    {
      v80[0] = (__int64)"switch.offset";
      v81 = 259;
      if ( *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(v27 + 16) > 0x10u )
        return sub_17D2EF0((__int64 *)a3, 11, (__int64 *)v6, v27, v80, 0, 0);
      else
        return (_QWORD *)sub_15A2B30((__int64 *)v6, v27, 0, 0, a4, a5, a6);
    }
  }
  else
  {
    return *(_QWORD **)(a1 + 8);
  }
  return (_QWORD *)v6;
}
