// Function: sub_1708AD0
// Address: 0x1708ad0
//
__int64 __fastcall sub_1708AD0(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r13
  __int64 *v6; // rbx
  char v7; // dl
  char v8; // cl
  __int64 v9; // r15
  char v10; // al
  unsigned int v11; // r12d
  char v12; // al
  unsigned int v13; // edx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // ecx
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rsi
  char v20; // bl
  unsigned int v21; // r13d
  unsigned __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 v24; // rcx
  unsigned int v25; // eax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdi
  unsigned int v30; // r10d
  unsigned int v31; // eax
  __int64 v32; // rdi
  unsigned int v33; // eax
  unsigned int v34; // r14d
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 *v37; // r8
  char *v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  __int64 v41; // rdi
  unsigned __int8 *v42; // rax
  unsigned __int8 **v43; // rdx
  unsigned __int8 *v44; // rsi
  unsigned __int64 v45; // rcx
  __int64 v46; // rcx
  __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // rdi
  unsigned __int64 v54; // rsi
  __int64 v55; // rsi
  __int64 v56; // r12
  __int64 v57; // r14
  unsigned int v58; // eax
  __int64 v59; // r15
  __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 v64; // rsi
  unsigned __int64 v65; // rcx
  __int64 v66; // rcx
  unsigned __int64 v67; // rsi
  _QWORD *v69; // r9
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  __int64 v74; // [rsp+10h] [rbp-A0h]
  unsigned int v75; // [rsp+18h] [rbp-98h]
  unsigned int v76; // [rsp+1Ch] [rbp-94h]
  __int64 v77; // [rsp+20h] [rbp-90h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  __int64 *v79; // [rsp+30h] [rbp-80h]
  __int64 *v80; // [rsp+30h] [rbp-80h]
  __int64 **v81; // [rsp+38h] [rbp-78h]
  __int64 v82; // [rsp+38h] [rbp-78h]
  __int64 v83; // [rsp+38h] [rbp-78h]
  char *v84; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v85; // [rsp+48h] [rbp-68h]
  char v86; // [rsp+50h] [rbp-60h]
  char v87; // [rsp+51h] [rbp-5Fh]
  __int64 v88; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v89; // [rsp+68h] [rbp-48h]
  __int64 v90; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v91; // [rsp+78h] [rbp-38h]

  v5 = a2;
  v6 = a1;
  v7 = *(_BYTE *)(a2 + 23);
  v77 = a2;
  v8 = v7 & 0x40;
  if ( (v7 & 0x40) != 0 )
  {
    v9 = **(_QWORD **)(a2 - 8);
    v10 = *(_BYTE *)(v9 + 16);
    if ( v10 != 35 )
    {
LABEL_3:
      if ( v10 != 5 )
        goto LABEL_5;
      if ( *(_WORD *)(v9 + 18) != 11 )
        goto LABEL_5;
      v83 = *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      if ( !v83 )
        goto LABEL_5;
      v56 = *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v56 + 16) != 13 )
        goto LABEL_5;
LABEL_70:
      v57 = 0;
      v58 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
      v59 = v58 - 1;
      if ( v58 != 1 )
      {
        do
        {
          ++v57;
          if ( (v7 & 0x40) != 0 )
            v60 = *(_QWORD *)(v5 - 8);
          else
            v60 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
          v61 = sub_15A2B60(*(__int64 **)(v60 + 24LL * (unsigned int)(2 * v57)), v56, 0, 0, a3, a4, a5);
          if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
            v62 = *(_QWORD *)(v5 - 8);
          else
            v62 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
          v63 = (__int64 *)(24LL * (unsigned int)(2 * v57) + v62);
          if ( *v63 )
          {
            v64 = v63[1];
            v65 = v63[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v65 = v64;
            if ( v64 )
              *(_QWORD *)(v64 + 16) = *(_QWORD *)(v64 + 16) & 3LL | v65;
          }
          *v63 = v61;
          if ( v61 )
          {
            v66 = *(_QWORD *)(v61 + 8);
            v63[1] = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = (unsigned __int64)(v63 + 1) | *(_QWORD *)(v66 + 16) & 3LL;
            v63[2] = (v61 + 8) | v63[2] & 3;
            *(_QWORD *)(v61 + 8) = v63;
          }
          v7 = *(_BYTE *)(v5 + 23);
        }
        while ( v59 != v57 );
        v8 = v7 & 0x40;
      }
      if ( v8 )
        v69 = *(_QWORD **)(v5 - 8);
      else
        v69 = (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      if ( *v69 )
      {
        v70 = v69[1];
        v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v71 = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
      }
      *v69 = v83;
      v72 = *(_QWORD *)(v83 + 8);
      v69[1] = v72;
      if ( v72 )
        *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
      v69[2] = v69[2] & 3LL | (v83 + 8);
      *(_QWORD *)(v83 + 8) = v69;
      return v77;
    }
  }
  else
  {
    v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v10 = *(_BYTE *)(v9 + 16);
    if ( v10 != 35 )
      goto LABEL_3;
  }
  v83 = *(_QWORD *)(v9 - 48);
  if ( v83 )
  {
    v56 = *(_QWORD *)(v9 - 24);
    if ( *(_BYTE *)(v56 + 16) == 13 )
      goto LABEL_70;
  }
LABEL_5:
  sub_14C2530((__int64)&v88, (__int64 *)v9, a1[333], 0, a1[330], a2, a1[332], 0);
  v76 = v89;
  if ( v89 > 0x40 )
  {
    v11 = sub_16A5810((__int64)&v88);
    goto LABEL_7;
  }
  v11 = 64;
  if ( v88 << (64 - (unsigned __int8)v89) == -1 )
  {
LABEL_7:
    v12 = v91;
    v75 = v91;
    if ( v91 <= 0x40 )
      goto LABEL_8;
    goto LABEL_87;
  }
  _BitScanReverse64(&v67, ~(v88 << (64 - (unsigned __int8)v89)));
  v12 = v91;
  v75 = v91;
  v11 = v67 ^ 0x3F;
  if ( v91 <= 0x40 )
  {
LABEL_8:
    v13 = 64;
    v14 = ~(v90 << (64 - v12));
    if ( v14 )
    {
      _BitScanReverse64(&v15, v14);
      v13 = v15 ^ 0x3F;
    }
    goto LABEL_10;
  }
LABEL_87:
  v13 = sub_16A5810((__int64)&v90);
LABEL_10:
  v16 = (*(_DWORD *)(v5 + 20) & 0xFFFFFFFu) >> 1;
  if ( v16 == 1 )
    goto LABEL_29;
  v74 = v9;
  v17 = 0;
  v18 = v16 - 1;
  v19 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
  v78 = v5;
  v20 = *(_BYTE *)(v5 + 23) & 0x40;
  v21 = v13;
  do
  {
    while ( 1 )
    {
      ++v17;
      v28 = v19;
      if ( v20 )
        v28 = *(_QWORD *)(v78 - 8);
      v29 = *(_QWORD *)(v28 + 24LL * (unsigned int)(2 * v17));
      v30 = *(_DWORD *)(v29 + 32);
      if ( v30 <= 0x40 )
        break;
      v31 = sub_16A57B0(v29 + 24);
      v32 = v29 + 24;
      if ( v11 > v31 )
        v11 = v31;
      v33 = sub_16A5810(v32);
      if ( v21 > v33 )
        v21 = v33;
      if ( v18 == v17 )
        goto LABEL_28;
    }
    v22 = *(_QWORD *)(v29 + 24);
    if ( !v22 )
    {
      v26 = -1;
      if ( v11 > v30 )
        v11 = *(_DWORD *)(v29 + 32);
LABEL_16:
      _BitScanReverse64(&v27, v26);
      v25 = v27 ^ 0x3F;
      goto LABEL_17;
    }
    _BitScanReverse64((unsigned __int64 *)&v23, v22);
    if ( v11 > ((unsigned int)v23 ^ 0x3F) + v30 - 64 )
      v11 = (v23 ^ 0x3F) + v30 - 64;
    v24 = v22 << (64 - (unsigned __int8)v30);
    v25 = 64;
    v26 = ~v24;
    if ( v26 )
      goto LABEL_16;
LABEL_17:
    if ( v21 > v25 )
      v21 = v25;
  }
  while ( v18 != v17 );
LABEL_28:
  v13 = v21;
  v9 = v74;
  v6 = a1;
  v5 = v78;
LABEL_29:
  if ( v13 < v11 )
    v13 = v11;
  v34 = v76 - v13;
  if ( v76 == v13 )
  {
LABEL_95:
    v77 = 0;
  }
  else
  {
    if ( v34 - 2 <= 5 )
    {
      v34 = 8;
      goto LABEL_34;
    }
    if ( v34 - 9 <= 6 )
    {
      v34 = 16;
    }
    else if ( v34 - 17 <= 0xE )
    {
      v34 = 32;
    }
    else if ( v34 - 33 <= 0x1E )
    {
      v34 = 64;
    }
    else if ( v34 > 0x40 )
    {
      goto LABEL_95;
    }
LABEL_34:
    if ( v34 >= v76 )
      goto LABEL_95;
    v35 = (_QWORD *)sub_16498A0(v5);
    v36 = sub_1644900(v35, v34);
    v37 = (__int64 *)v6[1];
    v81 = (__int64 **)v36;
    v37[1] = *(_QWORD *)(v5 + 40);
    v37[2] = v5 + 24;
    v38 = *(char **)(v5 + 48);
    v84 = v38;
    if ( v38 )
    {
      v79 = v37;
      sub_1623A60((__int64)&v84, (__int64)v38, 2);
      v37 = v79;
      v39 = *v79;
      if ( *v79 )
        goto LABEL_37;
LABEL_38:
      v40 = (unsigned __int8 *)v84;
      *v37 = (__int64)v84;
      if ( v40 )
      {
        sub_1623210((__int64)&v84, v40, (__int64)v37);
      }
      else if ( v84 )
      {
        sub_161E7C0((__int64)&v84, (__int64)v84);
      }
    }
    else
    {
      v39 = *v37;
      if ( *v37 )
      {
LABEL_37:
        v80 = v37;
        sub_161E7C0((__int64)v37, v39);
        v37 = v80;
        goto LABEL_38;
      }
    }
    v41 = v6[1];
    v87 = 1;
    v84 = "trunc";
    v86 = 3;
    v42 = sub_1708970(v41, 36, v9, v81, (__int64 *)&v84);
    if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      v43 = *(unsigned __int8 ***)(v5 - 8);
    else
      v43 = (unsigned __int8 **)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    if ( *v43 )
    {
      v44 = v43[1];
      v45 = (unsigned __int64)v43[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v45 = v44;
      if ( v44 )
        *((_QWORD *)v44 + 2) = *((_QWORD *)v44 + 2) & 3LL | v45;
    }
    *v43 = v42;
    if ( v42 )
    {
      v46 = *((_QWORD *)v42 + 1);
      v43[1] = (unsigned __int8 *)v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = (unsigned __int64)(v43 + 1) | *(_QWORD *)(v46 + 16) & 3LL;
      v43[2] = (unsigned __int8 *)((unsigned __int64)(v42 + 8) | (unsigned __int64)v43[2] & 3);
      *((_QWORD *)v42 + 1) = v43;
    }
    v47 = 0;
    v82 = ((*(_DWORD *)(v5 + 20) & 0xFFFFFFFu) >> 1) - 1;
    if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFFu) >> 1 != 1 )
    {
      do
      {
        ++v47;
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v48 = *(_QWORD *)(v5 - 8);
        else
          v48 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        sub_16A5A50((__int64)&v84, (__int64 *)(*(_QWORD *)(v48 + 24LL * (unsigned int)(2 * v47)) + 24LL), v34);
        v49 = (__int64 *)sub_16498A0(v5);
        v50 = sub_159C0E0(v49, (__int64)&v84);
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v51 = *(_QWORD *)(v5 - 8);
        else
          v51 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v52 = (__int64 *)(24LL * (unsigned int)(2 * v47) + v51);
        if ( *v52 )
        {
          v53 = v52[1];
          v54 = v52[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v54 = v53;
          if ( v53 )
            *(_QWORD *)(v53 + 16) = *(_QWORD *)(v53 + 16) & 3LL | v54;
        }
        *v52 = v50;
        if ( v50 )
        {
          v55 = *(_QWORD *)(v50 + 8);
          v52[1] = v55;
          if ( v55 )
            *(_QWORD *)(v55 + 16) = (unsigned __int64)(v52 + 1) | *(_QWORD *)(v55 + 16) & 3LL;
          v52[2] = (v50 + 8) | v52[2] & 3;
          *(_QWORD *)(v50 + 8) = v52;
        }
        if ( v85 > 0x40 && v84 )
          j_j___libc_free_0_0(v84);
      }
      while ( v82 != v47 );
    }
    v75 = v91;
  }
  if ( v75 > 0x40 && v90 )
    j_j___libc_free_0_0(v90);
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  return v77;
}
