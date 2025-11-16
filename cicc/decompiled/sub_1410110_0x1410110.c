// Function: sub_1410110
// Address: 0x1410110
//
__int64 __fastcall sub_1410110(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 *v8; // rcx
  __int64 v9; // rdi
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 *v16; // rax
  char v17; // dl
  __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rax
  _QWORD *v25; // r12
  __int64 v26; // rax
  _QWORD *v27; // r12
  __int64 v28; // rax
  __int64 *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 v33; // al
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 *v37; // rsi
  unsigned int v38; // edi
  __int64 *v39; // rcx
  __int64 v40; // rdx
  int v41; // ecx
  int v42; // r9d
  int v43; // r11d
  __int64 *v44; // r10
  int v45; // ecx
  int v46; // ecx
  int v47; // eax
  int v48; // edx
  __int64 v49; // rdi
  unsigned int v50; // eax
  __int64 v51; // rsi
  int v52; // r9d
  __int64 *v53; // r8
  int v54; // edx
  int v55; // edx
  __int64 v56; // rdi
  int v57; // r9d
  unsigned int v58; // eax
  __int64 v59; // rsi
  unsigned int v60; // [rsp+Ch] [rbp-114h]
  __int64 v61; // [rsp+18h] [rbp-108h] BYREF
  __int64 v62; // [rsp+20h] [rbp-100h] BYREF
  unsigned int v63; // [rsp+28h] [rbp-F8h]
  __int64 v64; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v65; // [rsp+38h] [rbp-E8h]
  __int64 *v66; // [rsp+40h] [rbp-E0h]
  __int64 v67; // [rsp+48h] [rbp-D8h]
  __int64 v68; // [rsp+50h] [rbp-D0h]
  __int64 v69; // [rsp+58h] [rbp-C8h] BYREF
  _BYTE v70[24]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+78h] [rbp-A8h]
  unsigned int v72; // [rsp+80h] [rbp-A0h]
  __int64 v73; // [rsp+90h] [rbp-90h]
  unsigned __int64 v74; // [rsp+98h] [rbp-88h]

  sub_140B840(
    (__int64)v70,
    *(_QWORD *)a1,
    *(_QWORD *)(a1 + 8),
    *(_QWORD *)(a1 + 16),
    *(unsigned __int8 *)(a1 + 248) << 8);
  sub_140E6D0((__int64)&v62, (__int64)v70, a2);
  if ( v63 > 1 && v65 > 1 )
  {
    sub_159C0E0(*(_QWORD *)(a1 + 16), &v64);
    v3 = sub_159C0E0(*(_QWORD *)(a1 + 16), &v62);
    goto LABEL_8;
  }
  v4 = sub_1649C60(a2);
  v5 = *(unsigned int *)(a1 + 136);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 120);
    v7 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (__int64 *)(v6 + 56LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_6:
      if ( v8 != (__int64 *)(v6 + 56 * v5) )
      {
        v3 = v8[3];
        goto LABEL_8;
      }
    }
    else
    {
      v41 = 1;
      while ( v9 != -8 )
      {
        v42 = v41 + 1;
        v7 = (v5 - 1) & (v41 + v7);
        v8 = (__int64 *)(v6 + 56LL * v7);
        v9 = *v8;
        if ( v4 == *v8 )
          goto LABEL_6;
        v41 = v42;
      }
    }
  }
  v11 = *(_QWORD *)(a1 + 32);
  v12 = *(_QWORD *)(a1 + 24);
  v66 = (__int64 *)(a1 + 24);
  v67 = v11;
  v13 = *(_QWORD *)(a1 + 40);
  v69 = v12;
  v68 = v13;
  if ( v12 )
    sub_1623A60(&v69, v12, 2);
  if ( *(_BYTE *)(v4 + 16) <= 0x17u )
    goto LABEL_28;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(v4 + 40);
  *(_QWORD *)(a1 + 40) = v4 + 24;
  v14 = *(_QWORD *)(v4 + 48);
  v61 = v14;
  if ( v14 )
  {
    sub_1623A60(&v61, v14, 2);
    if ( !*(_QWORD *)(a1 + 24) )
      goto LABEL_26;
    goto LABEL_25;
  }
  if ( *(_QWORD *)(a1 + 24) )
  {
LABEL_25:
    sub_161E7C0(a1 + 24);
LABEL_26:
    v15 = v61;
    *(_QWORD *)(a1 + 24) = v61;
    if ( v15 )
      sub_1623210(&v61, v15, a1 + 24);
  }
LABEL_28:
  v16 = *(__int64 **)(a1 + 152);
  if ( *(__int64 **)(a1 + 160) != v16 )
    goto LABEL_29;
  v37 = &v16[*(unsigned int *)(a1 + 172)];
  v38 = *(_DWORD *)(a1 + 172);
  if ( v16 != v37 )
  {
    v39 = 0;
    while ( v4 != *v16 )
    {
      if ( *v16 == -2 )
        v39 = v16;
      if ( v37 == ++v16 )
      {
        if ( !v39 )
          goto LABEL_87;
        *v39 = v4;
        --*(_DWORD *)(a1 + 176);
        ++*(_QWORD *)(a1 + 144);
        goto LABEL_57;
      }
    }
    goto LABEL_30;
  }
LABEL_87:
  if ( v38 < *(_DWORD *)(a1 + 168) )
  {
    *(_DWORD *)(a1 + 172) = v38 + 1;
    *v37 = v4;
    ++*(_QWORD *)(a1 + 144);
  }
  else
  {
LABEL_29:
    sub_16CCBA0(a1 + 144, v4);
    if ( !v17 )
    {
LABEL_30:
      v18 = 0;
      v3 = 0;
      goto LABEL_31;
    }
  }
LABEL_57:
  v33 = *(_BYTE *)(v4 + 16);
  if ( v33 > 0x17u )
  {
    if ( v33 == 56 )
    {
LABEL_60:
      v3 = sub_1410C60(a1, v4);
      v18 = v34;
      goto LABEL_31;
    }
    v3 = sub_1410040((__int64 *)a1, v4);
    v18 = v40;
  }
  else
  {
    v18 = 0;
    v3 = 0;
    if ( v33 == 5 && *(_WORD *)(v4 + 18) == 32 )
      goto LABEL_60;
  }
LABEL_31:
  v19 = *(_DWORD *)(a1 + 136);
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_110;
  }
  v20 = *(_QWORD *)(a1 + 120);
  v21 = (v19 - 1) & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
  v22 = (__int64 *)(v20 + 56LL * v21);
  v23 = *v22;
  if ( v4 != *v22 )
  {
    v43 = 1;
    v44 = 0;
    while ( v23 != -8 )
    {
      if ( !v44 && v23 == -16 )
        v44 = v22;
      v21 = (v19 - 1) & (v43 + v21);
      v22 = (__int64 *)(v20 + 56LL * v21);
      v23 = *v22;
      if ( v4 == *v22 )
        goto LABEL_33;
      ++v43;
    }
    v45 = *(_DWORD *)(a1 + 128);
    if ( v44 )
      v22 = v44;
    ++*(_QWORD *)(a1 + 112);
    v46 = v45 + 1;
    if ( 4 * v46 < 3 * v19 )
    {
      if ( v19 - *(_DWORD *)(a1 + 132) - v46 > v19 >> 3 )
        goto LABEL_104;
      v60 = ((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9);
      sub_140F200(a1 + 112, v19);
      v54 = *(_DWORD *)(a1 + 136);
      if ( v54 )
      {
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 120);
        v53 = 0;
        v57 = 1;
        v58 = v55 & v60;
        v22 = (__int64 *)(v56 + 56LL * (v55 & v60));
        v59 = *v22;
        v46 = *(_DWORD *)(a1 + 128) + 1;
        if ( v4 == *v22 )
          goto LABEL_104;
        while ( v59 != -8 )
        {
          if ( !v53 && v59 == -16 )
            v53 = v22;
          v58 = v55 & (v57 + v58);
          v22 = (__int64 *)(v56 + 56LL * v58);
          v59 = *v22;
          if ( v4 == *v22 )
            goto LABEL_104;
          ++v57;
        }
LABEL_114:
        if ( v53 )
          v22 = v53;
LABEL_104:
        *(_DWORD *)(a1 + 128) = v46;
        if ( *v22 != -8 )
          --*(_DWORD *)(a1 + 132);
        *v22 = v4;
        v25 = v22 + 1;
        v22[1] = 6;
        v22[2] = 0;
        v22[3] = 0;
        v22[4] = 6;
        v22[5] = 0;
        v22[6] = 0;
        if ( !v3 )
        {
          v27 = v22 + 4;
          if ( !v18 )
            goto LABEL_47;
          goto LABEL_44;
        }
        goto LABEL_37;
      }
      goto LABEL_136;
    }
LABEL_110:
    sub_140F200(a1 + 112, 2 * v19);
    v47 = *(_DWORD *)(a1 + 136);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a1 + 120);
      v50 = (v47 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v22 = (__int64 *)(v49 + 56LL * v50);
      v51 = *v22;
      v46 = *(_DWORD *)(a1 + 128) + 1;
      if ( v4 == *v22 )
        goto LABEL_104;
      v52 = 1;
      v53 = 0;
      while ( v51 != -8 )
      {
        if ( v51 == -16 && !v53 )
          v53 = v22;
        v50 = v48 & (v52 + v50);
        v22 = (__int64 *)(v49 + 56LL * v50);
        v51 = *v22;
        if ( v4 == *v22 )
          goto LABEL_104;
        ++v52;
      }
      goto LABEL_114;
    }
LABEL_136:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_33:
  v24 = v22[3];
  v25 = v22 + 1;
  if ( v3 != v24 )
  {
    if ( v24 != -8 && v24 != 0 && v24 != -16 )
      sub_1649B30(v22 + 1);
LABEL_37:
    v22[3] = v3;
    if ( v3 != 0 && v3 != -8 && v3 != -16 )
      sub_164C220(v25);
  }
  v26 = v22[6];
  v27 = v22 + 4;
  if ( v26 == v18 )
    goto LABEL_47;
  if ( v26 != 0 && v26 != -8 && v26 != -16 )
    sub_1649B30(v22 + 4);
LABEL_44:
  v22[6] = v18;
  if ( v18 != 0 && v18 != -8 && v18 != -16 )
    sub_164C220(v27);
LABEL_47:
  v28 = v67;
  v29 = v66;
  v30 = v68;
  if ( !v67 )
  {
    v66[1] = 0;
    v29[2] = 0;
    goto LABEL_62;
  }
  v66[1] = v67;
  v29[2] = v30;
  if ( v30 == v28 + 40 )
    goto LABEL_62;
  if ( !v30 )
    BUG();
  v31 = *(_QWORD *)(v30 + 24);
  v61 = v31;
  if ( v31 )
  {
    sub_1623A60(&v61, v31, 2);
    if ( !*v29 )
      goto LABEL_53;
  }
  else if ( !*v29 )
  {
    goto LABEL_62;
  }
  sub_161E7C0(v29);
LABEL_53:
  v32 = v61;
  *v29 = v61;
  if ( v32 )
  {
    sub_1623210(&v61, v32, v29);
    v29 = v66;
  }
  else
  {
    if ( v61 )
      sub_161E7C0(&v61);
    v29 = v66;
  }
LABEL_62:
  v61 = v69;
  if ( !v69 )
  {
    if ( v29 == &v61 )
      goto LABEL_8;
    if ( !*v29 )
      goto LABEL_66;
LABEL_73:
    sub_161E7C0(v29);
    goto LABEL_74;
  }
  sub_1623A60(&v61, v69, 2);
  if ( v29 == &v61 )
  {
LABEL_66:
    if ( v61 )
      sub_161E7C0(&v61);
    v35 = v69;
    goto LABEL_69;
  }
  if ( *v29 )
    goto LABEL_73;
LABEL_74:
  v36 = v61;
  *v29 = v61;
  if ( !v36 )
    goto LABEL_66;
  sub_1623210(&v61, v36, v29);
  v35 = v69;
LABEL_69:
  if ( v35 )
    sub_161E7C0(&v69);
LABEL_8:
  if ( v65 > 0x40 && v64 )
    j_j___libc_free_0_0(v64);
  if ( v63 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  if ( v74 != v73 )
    _libc_free(v74);
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  return v3;
}
