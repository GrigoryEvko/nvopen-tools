// Function: sub_1126070
// Address: 0x1126070
//
unsigned __int8 *__fastcall sub_1126070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  unsigned int v9; // ebx
  bool v10; // dl
  char *v11; // rcx
  char v12; // al
  unsigned int v13; // r8d
  unsigned int v14; // eax
  unsigned int v15; // r9d
  int v16; // eax
  unsigned __int64 v17; // rax
  int v18; // r9d
  bool v19; // al
  bool v20; // r12
  __int16 v21; // r13
  __int64 v22; // rbx
  _QWORD *v23; // r12
  _QWORD **v24; // rdx
  int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v30; // rsi
  unsigned int v31; // r8d
  unsigned __int64 v32; // rdi
  bool v33; // si
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // r9
  int v36; // ecx
  __int64 *v37; // rax
  __int64 v38; // rsi
  unsigned __int64 v39; // rax
  __int64 v40; // r13
  __int16 v41; // ax
  __int16 v42; // r14
  _QWORD *v43; // rax
  __int64 v44; // rax
  int v45; // eax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // rcx
  __int64 v50; // rax
  bool v51; // al
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // r9
  int v54; // eax
  bool v55; // al
  int v56; // ecx
  __int64 *v57; // rax
  int v58; // ecx
  __int64 *v59; // rax
  int v60; // eax
  unsigned int v61; // [rsp+8h] [rbp-98h]
  bool v62; // [rsp+Ch] [rbp-94h]
  bool v63; // [rsp+Ch] [rbp-94h]
  unsigned int v64; // [rsp+Ch] [rbp-94h]
  unsigned int v65; // [rsp+Ch] [rbp-94h]
  unsigned int v66; // [rsp+Ch] [rbp-94h]
  unsigned int v67; // [rsp+10h] [rbp-90h]
  int v68; // [rsp+10h] [rbp-90h]
  int v69; // [rsp+10h] [rbp-90h]
  unsigned int v70; // [rsp+10h] [rbp-90h]
  unsigned int v71; // [rsp+10h] [rbp-90h]
  int v72; // [rsp+10h] [rbp-90h]
  int v73; // [rsp+10h] [rbp-90h]
  unsigned int v74; // [rsp+10h] [rbp-90h]
  int v76; // [rsp+18h] [rbp-88h]
  int v77; // [rsp+18h] [rbp-88h]
  int v78; // [rsp+18h] [rbp-88h]
  __int64 v79; // [rsp+20h] [rbp-80h]
  __int64 v80; // [rsp+28h] [rbp-78h]
  __int64 v81; // [rsp+30h] [rbp-70h]
  __int64 v82; // [rsp+38h] [rbp-68h]
  unsigned __int64 v83; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v84; // [rsp+48h] [rbp-58h]
  __int16 v85; // [rsp+60h] [rbp-40h]

  v6 = a2;
  v9 = *(_DWORD *)(a5 + 8);
  if ( v9 <= 0x40 )
    v10 = *(_QWORD *)a5 == 0;
  else
    v10 = v9 == (unsigned int)sub_C444A0(a5);
  if ( v10 )
    return 0;
  v11 = *(char **)(a2 - 64);
  v12 = *v11;
  if ( (unsigned __int8)*v11 <= 0x1Cu )
  {
    if ( v12 != 5 || *((_WORD *)v11 + 1) != 27 )
    {
LABEL_6:
      v13 = *(_DWORD *)(a4 + 8);
      if ( v13 > 0x40 )
        goto LABEL_7;
      goto LABEL_46;
    }
  }
  else if ( v12 != 56 )
  {
    goto LABEL_6;
  }
  if ( !v9 )
    return 0;
  if ( v9 <= 0x40
     ? 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)a5
     : v9 == (unsigned int)sub_C445E0(a5) )
  {
    return 0;
  }
  v30 = *(_QWORD *)a5;
  if ( v9 > 0x40 )
    v30 = *(_QWORD *)(v30 + 8LL * ((v9 - 1) >> 6));
  v31 = *(_DWORD *)(a4 + 8);
  v32 = *(_QWORD *)a4;
  v33 = (v30 & (1LL << ((unsigned __int8)v9 - 1))) != 0;
  if ( v31 > 0x40 )
    v32 = *(_QWORD *)(v32 + 8LL * ((v31 - 1) >> 6));
  v70 = *(_DWORD *)(a4 + 8);
  if ( ((v32 & (1LL << ((unsigned __int8)v31 - 1))) != 0) != v33 )
    return 0;
  a2 = a4;
  if ( (int)sub_C4C880(a5, a4) > 0 )
    return 0;
  v13 = v70;
  v10 = 1;
  if ( v70 > 0x40 )
  {
LABEL_7:
    v62 = v10;
    v67 = v13;
    if ( v13 != (unsigned int)sub_C444A0(a4) )
    {
      a2 = a5;
      if ( !sub_C43C50(a4, (const void **)a5) )
      {
        if ( v62 && (*(_QWORD *)(*(_QWORD *)a4 + 8LL * ((v67 - 1) >> 6)) & (1LL << ((unsigned __int8)v67 - 1))) != 0 )
        {
          v45 = sub_C44500(a4);
          v13 = v67;
          v18 = v45;
LABEL_73:
          if ( v9 <= 0x40 )
          {
            if ( v9 )
            {
              if ( *(_QWORD *)a5 << (64 - (unsigned __int8)v9) == -1 )
              {
                v18 -= 64;
              }
              else
              {
                _BitScanReverse64(&v46, ~(*(_QWORD *)a5 << (64 - (unsigned __int8)v9)));
                v18 -= v46 ^ 0x3F;
              }
            }
            if ( v18 <= 0 )
              goto LABEL_68;
            v84 = v9;
            goto LABEL_79;
          }
          v18 -= sub_C44500(a5);
          if ( v18 <= 0 )
            goto LABEL_68;
          v84 = v9;
          goto LABEL_96;
        }
        v14 = sub_C444A0(a4);
        v10 = v62;
        v13 = v67;
        v15 = v14;
        goto LABEL_11;
      }
LABEL_54:
      v21 = 32;
      v22 = sub_AD6530(*(_QWORD *)(a3 + 8), a2);
      if ( (*(_WORD *)(v6 + 2) & 0x3F) == 0x21 )
        v21 = sub_B52870(32);
      v85 = 257;
      v23 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v23 )
        return (unsigned __int8 *)v23;
      v24 = *(_QWORD ***)(a3 + 8);
      v36 = *((unsigned __int8 *)v24 + 8);
      if ( (unsigned int)(v36 - 17) <= 1 )
      {
        BYTE4(v79) = (_BYTE)v36 == 18;
        LODWORD(v79) = *((_DWORD *)v24 + 8);
        v37 = (__int64 *)sub_BCB2A0(*v24);
        v27 = sub_BCE1B0(v37, v79);
        goto LABEL_59;
      }
LABEL_67:
      v27 = sub_BCB2A0(*v24);
      goto LABEL_59;
    }
LABEL_60:
    if ( v9 > 0x40 )
    {
      v38 = v9 - 1 - (unsigned int)sub_C444A0(a5);
    }
    else
    {
      v38 = 0xFFFFFFFFLL;
      if ( *(_QWORD *)a5 )
      {
        _BitScanReverse64(&v39, *(_QWORD *)a5);
        v38 = 63 - ((unsigned int)v39 ^ 0x3F);
      }
    }
    v40 = sub_AD64C0(*(_QWORD *)(a3 + 8), v38, 0);
    v41 = *(_WORD *)(v6 + 2);
    v42 = 34;
    if ( (v41 & 0x3F) == 0x21 )
      v42 = sub_B52870(34);
    v85 = 257;
    v43 = sub_BD2C40(72, unk_3F10FD0);
    v23 = v43;
    if ( v43 )
      sub_1113300((__int64)v43, v42, a3, v40, (__int64)&v83);
    return (unsigned __int8 *)v23;
  }
LABEL_46:
  if ( !*(_QWORD *)a4 )
    goto LABEL_60;
  v34 = *(_QWORD *)a4;
  if ( *(_QWORD *)a4 == *(_QWORD *)a5 )
    goto LABEL_54;
  if ( v10 && _bittest64((const __int64 *)&v34, v13 - 1) )
  {
    if ( v13 )
    {
      v18 = 64;
      v52 = ~(v34 << (64 - (unsigned __int8)v13));
      if ( v52 )
      {
        _BitScanReverse64(&v53, v52);
        v18 = v53 ^ 0x3F;
      }
    }
    else
    {
      v18 = 0;
    }
    goto LABEL_73;
  }
  v15 = v13;
  if ( v34 )
  {
    _BitScanReverse64(&v35, v34);
    v15 = v13 - 64 + (v35 ^ 0x3F);
  }
LABEL_11:
  if ( v9 > 0x40 )
  {
    v61 = v15;
    v63 = v10;
    v71 = v13;
    v16 = sub_C444A0(a5);
    v15 = v61;
    v10 = v63;
    v13 = v71;
  }
  else
  {
    v16 = v9;
    if ( *(_QWORD *)a5 )
    {
      _BitScanReverse64(&v17, *(_QWORD *)a5);
      v16 = v9 - 64 + (v17 ^ 0x3F);
    }
  }
  v18 = v15 - v16;
  if ( v18 <= 0 )
    goto LABEL_68;
  if ( v10 )
  {
    v84 = v9;
    if ( v9 <= 0x40 )
    {
LABEL_79:
      v47 = *(_QWORD *)a5;
      goto LABEL_80;
    }
LABEL_96:
    v66 = v18;
    sub_C43780((__int64)&v83, (const void **)a5);
    v9 = v84;
    v18 = v66;
    if ( v84 > 0x40 )
    {
      sub_C44B70((__int64)&v83, v66);
      v13 = *(_DWORD *)(a4 + 8);
      v9 = v84;
      v18 = v66;
      goto LABEL_84;
    }
    v47 = v83;
    v13 = *(_DWORD *)(a4 + 8);
LABEL_80:
    v48 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
    v49 = 0;
    if ( v9 )
    {
      v50 = (__int64)(v47 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
      if ( v9 == v18 )
        v49 = v48 & (v50 >> 63);
      else
        v49 = v48 & (v50 >> v18);
    }
    v83 = v49;
LABEL_84:
    if ( v13 <= 0x40 )
    {
      if ( *(_QWORD *)a4 != v83 )
        goto LABEL_86;
    }
    else
    {
      v64 = v13;
      v72 = v18;
      v51 = sub_C43C50(a4, (const void **)&v83);
      v18 = v72;
      v13 = v64;
      if ( !v51 )
      {
LABEL_86:
        if ( v9 > 0x40 && v83 )
        {
          v73 = v18;
          j_j___libc_free_0_0(v83);
          v9 = *(_DWORD *)(a5 + 8);
          v18 = v73;
        }
        else
        {
          v9 = *(_DWORD *)(a5 + 8);
        }
        goto LABEL_16;
      }
    }
    if ( v9 > 0x40 && v83 )
    {
      v78 = v18;
      j_j___libc_free_0_0(v83);
      v13 = *(_DWORD *)(a4 + 8);
      v18 = v78;
    }
    if ( v13 )
    {
      if ( v13 <= 0x40 )
      {
        v55 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *(_QWORD *)a4;
      }
      else
      {
        v74 = v13;
        v76 = v18;
        v54 = sub_C445E0(a4);
        v18 = v76;
        v55 = v54 == v74;
      }
      if ( !v55 )
      {
LABEL_104:
        v21 = 32;
        v22 = sub_AD64C0(*(_QWORD *)(a3 + 8), v18, 0);
        if ( (*(_WORD *)(v6 + 2) & 0x3F) == 0x21 )
          v21 = sub_B52870(32);
        v85 = 257;
        v23 = sub_BD2C40(72, unk_3F10FD0);
        if ( !v23 )
          return (unsigned __int8 *)v23;
        v24 = *(_QWORD ***)(a3 + 8);
        v56 = *((unsigned __int8 *)v24 + 8);
        if ( (unsigned int)(v56 - 17) <= 1 )
        {
          BYTE4(v80) = (_BYTE)v56 == 18;
          LODWORD(v80) = *((_DWORD *)v24 + 8);
          v57 = (__int64 *)sub_BCB2A0(*v24);
          v27 = sub_BCE1B0(v57, v80);
          goto LABEL_59;
        }
        goto LABEL_67;
      }
    }
    if ( *(_DWORD *)(a5 + 8) > 0x40u )
    {
      v77 = v18;
      v60 = sub_C44630(a5);
      v18 = v77;
      if ( v60 == 1 )
        goto LABEL_104;
    }
    else if ( *(_QWORD *)a5 && (*(_QWORD *)a5 & (*(_QWORD *)a5 - 1LL)) == 0 )
    {
      goto LABEL_104;
    }
    v21 = 35;
    v22 = sub_AD64C0(*(_QWORD *)(a3 + 8), v18, 0);
    if ( (*(_WORD *)(v6 + 2) & 0x3F) == 0x21 )
      v21 = sub_B52870(35);
    v85 = 257;
    v23 = sub_BD2C40(72, unk_3F10FD0);
    if ( !v23 )
      return (unsigned __int8 *)v23;
    v24 = *(_QWORD ***)(a3 + 8);
    v58 = *((unsigned __int8 *)v24 + 8);
    if ( (unsigned int)(v58 - 17) <= 1 )
    {
      BYTE4(v81) = (_BYTE)v58 == 18;
      LODWORD(v81) = *((_DWORD *)v24 + 8);
      v59 = (__int64 *)sub_BCB2A0(*v24);
      v27 = sub_BCE1B0(v59, v81);
      goto LABEL_59;
    }
    goto LABEL_67;
  }
LABEL_16:
  v84 = v9;
  if ( v9 <= 0x40 )
  {
    v83 = *(_QWORD *)a5;
    goto LABEL_18;
  }
  v65 = v18;
  sub_C43780((__int64)&v83, (const void **)a5);
  v9 = v84;
  v18 = v65;
  if ( v84 <= 0x40 )
  {
LABEL_18:
    if ( v18 == v9 )
      v83 = 0;
    else
      v83 >>= v18;
    goto LABEL_20;
  }
  sub_C482E0((__int64)&v83, v65);
  v18 = v65;
LABEL_20:
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
  {
    v20 = *(_QWORD *)a4 == v83;
  }
  else
  {
    v68 = v18;
    v19 = sub_C43C50(a4, (const void **)&v83);
    v18 = v68;
    v20 = v19;
  }
  if ( v84 > 0x40 && v83 )
  {
    v69 = v18;
    j_j___libc_free_0_0(v83);
    v18 = v69;
  }
  if ( v20 )
  {
    v21 = 32;
    v22 = sub_AD64C0(*(_QWORD *)(a3 + 8), v18, 0);
    if ( (*(_WORD *)(v6 + 2) & 0x3F) == 0x21 )
      v21 = sub_B52870(32);
    v85 = 257;
    v23 = sub_BD2C40(72, unk_3F10FD0);
    if ( !v23 )
      return (unsigned __int8 *)v23;
    v24 = *(_QWORD ***)(a3 + 8);
    v25 = *((unsigned __int8 *)v24 + 8);
    if ( (unsigned int)(v25 - 17) <= 1 )
    {
      BYTE4(v82) = (_BYTE)v25 == 18;
      LODWORD(v82) = *((_DWORD *)v24 + 8);
      v26 = (__int64 *)sub_BCB2A0(*v24);
      v27 = sub_BCE1B0(v26, v82);
LABEL_59:
      sub_B523C0((__int64)v23, v27, 53, v21, a3, v22, (__int64)&v83, 0, 0, 0);
      return (unsigned __int8 *)v23;
    }
    goto LABEL_67;
  }
LABEL_68:
  v44 = sub_AD64C0(*(_QWORD *)(v6 + 8), (*(_WORD *)(v6 + 2) & 0x3F) == 33, 0);
  return sub_F162A0(a1, v6, v44);
}
