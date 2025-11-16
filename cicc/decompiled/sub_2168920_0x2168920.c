// Function: sub_2168920
// Address: 0x2168920
//
__int64 __fastcall sub_2168920(__int64 *a1, __int64 a2, _DWORD *a3)
{
  _BYTE *v4; // rax
  __int64 v5; // rsi
  bool (__fastcall *v6)(__int64, __int64); // rax
  _DWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r13
  bool v16; // al
  unsigned int v18; // ecx
  unsigned int v19; // ecx
  int v20; // eax
  __int64 *v21; // r8
  unsigned int v22; // eax
  __int64 v23; // rbx
  __int64 *v24; // r9
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 *v28; // rdi
  __int64 *v29; // rax
  __int64 *v30; // rcx
  int v31; // ebx
  unsigned __int64 v32; // r13
  unsigned int v33; // r14d
  unsigned __int64 *v34; // r15
  unsigned int v35; // ebx
  unsigned __int64 *v36; // r13
  unsigned __int64 v37; // rbx
  bool (__fastcall *v38)(__int64, __int64, __int64, unsigned __int64); // rax
  __int64 v39; // r12
  unsigned int v40; // r12d
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // r14
  __int64 v44; // [rsp+8h] [rbp-E8h]
  _BYTE *v45; // [rsp+10h] [rbp-E0h]
  char v46; // [rsp+1Bh] [rbp-D5h]
  unsigned int v47; // [rsp+1Ch] [rbp-D4h]
  __int64 v48; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v49; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v50; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v51; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v52; // [rsp+48h] [rbp-A8h]
  unsigned __int64 *v53; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-98h]
  unsigned __int64 *v55; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v56; // [rsp+68h] [rbp-88h]
  _QWORD *v57; // [rsp+70h] [rbp-80h] BYREF
  __int64 *v58; // [rsp+78h] [rbp-78h]
  __int64 *v59; // [rsp+80h] [rbp-70h]
  __int64 v60; // [rsp+88h] [rbp-68h]
  int v61; // [rsp+90h] [rbp-60h]
  _BYTE v62[88]; // [rsp+98h] [rbp-58h] BYREF

  v44 = *a1;
  v47 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
  v4 = (_BYTE *)a1[2];
  *a3 = 0;
  v45 = v4;
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
  v6 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 360LL);
  if ( v6 == sub_1F3D0C0 )
  {
    v57 = (_QWORD *)sub_1560340((_QWORD *)(v5 + 112), -1, "no-jump-tables", 0xEu);
    v7 = (_DWORD *)sub_155D8B0((__int64 *)&v57);
    if ( v8 == 4 && *v7 == 1702195828 )
    {
      if ( !v47 )
        return 0;
LABEL_20:
      if ( 8 * (unsigned int)sub_15A95A0(v44, 0) < v47 )
        return v47;
      v46 = 0;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
        goto LABEL_22;
LABEL_6:
      v9 = *(_QWORD *)(a2 - 8);
      goto LABEL_7;
    }
    if ( (v45[2871] & 0xFB) != 0 && (v45[2870] & 0xFB) != 0 )
    {
      if ( v47 )
        goto LABEL_20;
      return 0;
    }
    if ( !v47 )
      return 0;
  }
  else
  {
    v16 = v6((__int64)v45, v5);
    if ( !v47 )
      return 0;
    if ( !v16 )
      goto LABEL_20;
  }
  v46 = 1;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    goto LABEL_6;
LABEL_22:
  v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
LABEL_7:
  v10 = *(_QWORD *)(v9 + 48);
  v11 = *(_DWORD *)(v10 + 32);
  v50 = v11;
  if ( v11 <= 0x40 )
  {
    v12 = *(unsigned __int64 **)(v10 + 24);
    v52 = v11;
    v49 = v12;
LABEL_9:
    v51 = (unsigned __int64)v49;
    goto LABEL_10;
  }
  sub_16A4FD0((__int64)&v49, (const void **)(v10 + 24));
  v52 = v50;
  if ( v50 <= 0x40 )
    goto LABEL_9;
  sub_16A4FD0((__int64)&v51, (const void **)&v49);
LABEL_10:
  v13 = 0;
  v48 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 != 1 )
  {
    while ( 1 )
    {
      ++v13;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v14 = *(_QWORD *)(a2 - 8);
      else
        v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v15 = *(_QWORD *)(v14 + 24LL * (unsigned int)(2 * v13));
      if ( (int)sub_16AEA10(v15 + 24, (__int64)&v49) > 0 )
      {
        if ( v50 <= 0x40 && (v18 = *(_DWORD *)(v15 + 32), v18 <= 0x40) )
        {
          v50 = *(_DWORD *)(v15 + 32);
          v49 = (unsigned __int64 *)(*(_QWORD *)(v15 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18));
        }
        else
        {
          sub_16A51C0((__int64)&v49, v15 + 24);
        }
      }
      if ( (int)sub_16AEA10(v15 + 24, (__int64)&v51) >= 0 )
        goto LABEL_15;
      if ( v52 <= 0x40 && (v19 = *(_DWORD *)(v15 + 32), v19 <= 0x40) )
      {
        v52 = *(_DWORD *)(v15 + 32);
        v51 = *(_QWORD *)(v15 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v19);
LABEL_15:
        if ( v48 == v13 )
          break;
      }
      else
      {
        sub_16A51C0((__int64)&v51, v15 + 24);
        if ( v48 == v13 )
          break;
      }
    }
  }
  if ( v47 <= 8 * (unsigned int)sub_15A95A0(v44, 0) )
  {
    v20 = *(_DWORD *)(a2 + 20);
    v21 = (__int64 *)v62;
    v57 = 0;
    v58 = (__int64 *)v62;
    v59 = (__int64 *)v62;
    v22 = (v20 & 0xFFFFFFFu) >> 1;
    v60 = 4;
    v23 = v22 - 1;
    v61 = 0;
    if ( v22 != 1 )
    {
      v24 = (__int64 *)v62;
      v25 = 0;
      while ( 1 )
      {
LABEL_38:
        v27 = 24;
        if ( (_DWORD)v25 != -2 )
          v27 = 24LL * (unsigned int)(2 * v25 + 3);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        {
          v26 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v27);
          if ( v21 != v24 )
            goto LABEL_36;
        }
        else
        {
          v26 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) + v27);
          if ( v21 != v24 )
            goto LABEL_36;
        }
        v28 = &v21[HIDWORD(v60)];
        if ( v28 != v21 )
        {
          v29 = v21;
          v30 = 0;
          while ( v26 != *v29 )
          {
            if ( *v29 == -2 )
              v30 = v29;
            if ( v28 == ++v29 )
            {
              if ( !v30 )
                goto LABEL_93;
              ++v25;
              *v30 = v26;
              v24 = v59;
              --v61;
              v21 = v58;
              v57 = (_QWORD *)((char *)v57 + 1);
              if ( v23 != v25 )
                goto LABEL_38;
              goto LABEL_50;
            }
          }
          goto LABEL_37;
        }
LABEL_93:
        if ( HIDWORD(v60) < (unsigned int)v60 )
        {
          ++HIDWORD(v60);
          *v28 = v26;
          v21 = v58;
          v57 = (_QWORD *)((char *)v57 + 1);
          v24 = v59;
          goto LABEL_37;
        }
LABEL_36:
        sub_16CCBA0((__int64)&v57, v26);
        v24 = v59;
        v21 = v58;
LABEL_37:
        if ( v23 == ++v25 )
        {
LABEL_50:
          v31 = HIDWORD(v60) - v61;
          goto LABEL_51;
        }
      }
    }
    v31 = 0;
LABEL_51:
    v32 = 8 * (unsigned int)sub_15A95A0(v44, 0);
    v56 = v50;
    if ( v50 > 0x40 )
      sub_16A4FD0((__int64)&v55, (const void **)&v49);
    else
      v55 = v49;
    sub_16A7590((__int64)&v55, (__int64 *)&v51);
    v33 = v56;
    v34 = v55;
    v56 = 0;
    v54 = v33;
    v53 = v55;
    if ( v33 <= 0x40 )
    {
      if ( v55 == (unsigned __int64 *)-1LL )
        goto LABEL_55;
      v42 = (unsigned __int64)v55 + 1;
LABEL_101:
      if ( v32 >= v42 && (v31 == 1 && v47 > 2 || v31 == 2 && v47 > 4 || v31 == 3 && v47 > 5) )
      {
        if ( v59 != v58 )
          _libc_free((unsigned __int64)v59);
        goto LABEL_76;
      }
      goto LABEL_55;
    }
    if ( v33 - (unsigned int)sub_16A57B0((__int64)&v53) <= 0x40 )
    {
      v42 = *v34;
      if ( *v34 == -1 )
      {
        j_j___libc_free_0_0(v34);
        if ( v56 <= 0x40 )
        {
LABEL_55:
          if ( v59 != v58 )
            _libc_free((unsigned __int64)v59);
          goto LABEL_57;
        }
LABEL_99:
        if ( v55 )
          j_j___libc_free_0_0(v55);
        goto LABEL_101;
      }
      ++v42;
    }
    else
    {
      v42 = -1;
    }
    if ( !v34 )
      goto LABEL_101;
    j_j___libc_free_0_0(v34);
    if ( v56 <= 0x40 )
      goto LABEL_101;
    goto LABEL_99;
  }
LABEL_57:
  if ( !v46 || v47 == 1 || (*(unsigned int (__fastcall **)(_BYTE *))(*(_QWORD *)v45 + 472LL))(v45) > v47 )
    goto LABEL_77;
  v56 = v50;
  if ( v50 > 0x40 )
    sub_16A4FD0((__int64)&v55, (const void **)&v49);
  else
    v55 = v49;
  sub_16A7590((__int64)&v55, (__int64 *)&v51);
  v35 = v56;
  v36 = v55;
  v56 = 0;
  LODWORD(v58) = v35;
  v57 = v55;
  if ( v35 <= 0x40 )
  {
    if ( v55 == (unsigned __int64 *)-1LL )
      v37 = -1;
    else
      v37 = (unsigned __int64)v55 + 1;
    goto LABEL_69;
  }
  if ( v35 - (unsigned int)sub_16A57B0((__int64)&v57) > 0x40 )
  {
    v37 = -1;
    goto LABEL_65;
  }
  v37 = *v36;
  if ( *v36 != -1 )
  {
    ++v37;
LABEL_65:
    if ( !v36 )
      goto LABEL_69;
  }
  j_j___libc_free_0_0(v36);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
LABEL_69:
  v38 = *(bool (__fastcall **)(__int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v45 + 368LL);
  if ( v38 == sub_1F44290 )
  {
    v39 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) + 112LL;
    if ( (unsigned __int8)sub_1560180(v39, 34) || (unsigned __int8)sub_1560180(v39, 17) )
    {
      v40 = sub_1F44230((__int64)v45, 1);
    }
    else
    {
      v40 = sub_1F44230((__int64)v45, 0);
      if ( (unsigned int)sub_1F44280(v45) )
      {
        v41 = (unsigned int)sub_1F44280(v45);
        goto LABEL_73;
      }
    }
    v41 = 0xFFFFFFFFLL;
LABEL_73:
    if ( v37 > v41 || 100 * (unsigned __int64)v47 < v37 * v40 )
      goto LABEL_77;
    goto LABEL_75;
  }
  if ( v38((__int64)v45, a2, v47, v37) )
  {
LABEL_75:
    *a3 = v37;
LABEL_76:
    v47 = 1;
  }
LABEL_77:
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  return v47;
}
