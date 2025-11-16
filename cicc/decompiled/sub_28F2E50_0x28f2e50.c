// Function: sub_28F2E50
// Address: 0x28f2e50
//
_BYTE *__fastcall sub_28F2E50(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, _BYTE *a4)
{
  int v4; // eax
  __int64 v5; // rdx
  _BYTE *v6; // rbx
  __int64 v7; // rdi
  unsigned __int8 v8; // al
  __int64 *v9; // r15
  unsigned int *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // r14
  char v15; // al
  unsigned __int8 **v16; // rsi
  unsigned __int8 *v17; // rdi
  unsigned __int8 v18; // cl
  __int64 *v19; // r15
  unsigned int *v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rsi
  _BYTE *v25; // rax
  unsigned __int8 **v26; // rsi
  _BYTE *v27; // rax
  _BYTE *v28; // rdi
  _BYTE *v29; // r13
  __int64 v31; // rcx
  _QWORD *v32; // r15
  _QWORD *v33; // r12
  _QWORD *i; // r13
  _QWORD *k; // r13
  _QWORD *j; // r14
  unsigned int *v37; // rax
  unsigned int *v38; // rax
  char v39; // r8
  __int64 v40; // rax
  char v41; // [rsp+17h] [rbp-169h]
  _BYTE *v46; // [rsp+40h] [rbp-140h]
  __int64 v47; // [rsp+58h] [rbp-128h]
  unsigned int *v48; // [rsp+60h] [rbp-120h] BYREF
  __int64 v49; // [rsp+68h] [rbp-118h]
  char v50; // [rsp+74h] [rbp-10Ch]
  __int16 v51; // [rsp+80h] [rbp-100h]
  _BYTE *v52; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v53; // [rsp+98h] [rbp-E8h]
  _BYTE v54[32]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int *v55; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+C8h] [rbp-B8h]
  _BYTE v57[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+F0h] [rbp-90h]
  __int64 v59; // [rsp+F8h] [rbp-88h]
  __int16 v60; // [rsp+100h] [rbp-80h]
  __int64 v61; // [rsp+108h] [rbp-78h]
  void **v62; // [rsp+110h] [rbp-70h]
  void **v63; // [rsp+118h] [rbp-68h]
  __int64 v64; // [rsp+120h] [rbp-60h]
  int v65; // [rsp+128h] [rbp-58h]
  __int16 v66; // [rsp+12Ch] [rbp-54h]
  char v67; // [rsp+12Eh] [rbp-52h]
  __int64 v68; // [rsp+130h] [rbp-50h]
  __int64 v69; // [rsp+138h] [rbp-48h]
  void *v70; // [rsp+140h] [rbp-40h] BYREF
  void *v71; // [rsp+148h] [rbp-38h] BYREF

  v52 = v54;
  v53 = 0x400000000LL;
  sub_28ED0D0(a3, (__int64)&v52);
  v4 = v53;
  if ( !(_DWORD)v53 )
    goto LABEL_112;
  v5 = *a2;
  v41 = v5;
  if ( (_BYTE)v5 != 45 )
  {
    if ( (v53 & 1) == 0 )
    {
      v6 = v52;
      v46 = &v52[8 * (unsigned int)v53];
      goto LABEL_27;
    }
    v39 = sub_28EECD0((__int64)a2);
    v4 = v53;
    if ( v39 )
    {
LABEL_112:
      v28 = v52;
      v29 = 0;
      goto LABEL_71;
    }
  }
  v6 = v52;
  v5 = (__int64)&v52[8 * v4];
  v46 = (_BYTE *)v5;
  if ( (_BYTE *)v5 == v52 )
    goto LABEL_69;
  do
  {
LABEL_27:
    v14 = *(_QWORD *)v6;
    v15 = *(_BYTE *)(*(_QWORD *)v6 + 7LL) & 0x40;
    if ( v15 )
    {
      v16 = *(unsigned __int8 ***)(v14 - 8);
      v17 = *v16;
      v18 = **v16;
      v19 = (__int64 *)(*v16 + 24);
      if ( v18 == 18 )
        goto LABEL_29;
    }
    else
    {
      v26 = (unsigned __int8 **)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
      v17 = *v26;
      v18 = **v26;
      v19 = (__int64 *)(*v26 + 24);
      if ( v18 == 18 )
        goto LABEL_29;
    }
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v17 + 1) + 8LL) - 17 <= 1 && v18 <= 0x15u )
    {
      v27 = sub_AD7630((__int64)v17, 0, v5);
      if ( !v27 || *v27 != 18 )
        goto LABEL_48;
      v19 = (__int64 *)(v27 + 24);
LABEL_29:
      v20 = (unsigned int *)sub_C33340();
      if ( (unsigned int *)*v19 == v20 )
        sub_C3C790(&v48, (_QWORD **)v19);
      else
        sub_C33EB0(&v48, v19);
      if ( v20 == v48 )
      {
        if ( (*(_BYTE *)(v49 + 20) & 8) == 0 )
          goto LABEL_67;
        sub_C3CCB0((__int64)&v48);
        v37 = v48;
      }
      else
      {
        if ( (v50 & 8) == 0 )
        {
LABEL_33:
          sub_C338E0((__int64)&v55, (__int64)&v48);
          goto LABEL_34;
        }
        sub_C34440((unsigned __int8 *)&v48);
        v37 = v48;
      }
      if ( v20 != v37 )
        goto LABEL_33;
LABEL_67:
      sub_C3C840(&v55, &v48);
LABEL_34:
      v21 = sub_AD8F10(*(_QWORD *)(v14 + 8), (__int64 *)&v55);
      if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
        v22 = *(_QWORD *)(v14 - 8);
      else
        v22 = v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v22 )
      {
        v23 = *(_QWORD *)(v22 + 8);
        **(_QWORD **)(v22 + 16) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
      }
      *(_QWORD *)v22 = v21;
      if ( v21 )
      {
        v24 = *(_QWORD *)(v21 + 16);
        *(_QWORD *)(v22 + 8) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = v22 + 8;
        *(_QWORD *)(v22 + 16) = v21 + 16;
        *(_QWORD *)(v21 + 16) = v22;
      }
      if ( v55 == v20 )
      {
        if ( v56 )
        {
          v31 = *(_QWORD *)(v56 - 8);
          v32 = (_QWORD *)(v56 + 24 * v31);
          if ( (_QWORD *)v56 != v32 )
          {
            v33 = (_QWORD *)(v56 + 24 * v31);
            do
            {
              v33 -= 3;
              sub_91D830(v33);
            }
            while ( (_QWORD *)v56 != v33 );
            v32 = v33;
          }
          j_j_j___libc_free_0_0((unsigned __int64)(v32 - 1));
          if ( v48 != v20 )
            goto LABEL_46;
          goto LABEL_80;
        }
      }
      else
      {
        sub_C338F0((__int64)&v55);
      }
      if ( v48 != v20 )
      {
LABEL_46:
        sub_C338F0((__int64)&v48);
LABEL_47:
        *(_BYTE *)(a1 + 752) = 1;
LABEL_48:
        v15 = *(_BYTE *)(v14 + 7) & 0x40;
        goto LABEL_49;
      }
LABEL_80:
      if ( v49 )
      {
        for ( i = (_QWORD *)(v49 + 24LL * *(_QWORD *)(v49 - 8)); (_QWORD *)v49 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      }
      goto LABEL_47;
    }
LABEL_49:
    if ( v15 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)(v14 - 8) + 32LL);
      v8 = *(_BYTE *)v7;
      v9 = (__int64 *)(v7 + 24);
      if ( *(_BYTE *)v7 == 18 )
        goto LABEL_7;
    }
    else
    {
      v7 = *(_QWORD *)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF) + 32);
      v8 = *(_BYTE *)v7;
      v9 = (__int64 *)(v7 + 24);
      if ( *(_BYTE *)v7 == 18 )
        goto LABEL_7;
    }
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
      goto LABEL_26;
    if ( v8 > 0x15u )
      goto LABEL_26;
    v25 = sub_AD7630(v7, 0, v5);
    if ( !v25 || *v25 != 18 )
      goto LABEL_26;
    v9 = (__int64 *)(v25 + 24);
LABEL_7:
    v10 = (unsigned int *)sub_C33340();
    if ( (unsigned int *)*v9 == v10 )
      sub_C3C790(&v48, (_QWORD **)v9);
    else
      sub_C33EB0(&v48, v9);
    if ( v10 == v48 )
    {
      if ( (*(_BYTE *)(v49 + 20) & 8) == 0 )
        goto LABEL_65;
      sub_C3CCB0((__int64)&v48);
      v38 = v48;
    }
    else
    {
      if ( (v50 & 8) == 0 )
        goto LABEL_11;
      sub_C34440((unsigned __int8 *)&v48);
      v38 = v48;
    }
    if ( v10 == v38 )
    {
LABEL_65:
      sub_C3C840(&v55, &v48);
      goto LABEL_12;
    }
LABEL_11:
    sub_C338E0((__int64)&v55, (__int64)&v48);
LABEL_12:
    v11 = sub_AD8F10(*(_QWORD *)(v14 + 8), (__int64 *)&v55);
    if ( (*(_BYTE *)(v14 + 7) & 0x40) != 0 )
      v5 = *(_QWORD *)(v14 - 8);
    else
      v5 = v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)(v5 + 32) )
    {
      v12 = *(_QWORD *)(v5 + 40);
      **(_QWORD **)(v5 + 48) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *(_QWORD *)(v5 + 48);
    }
    *(_QWORD *)(v5 + 32) = v11;
    if ( v11 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      *(_QWORD *)(v5 + 40) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = v5 + 40;
      *(_QWORD *)(v5 + 48) = v11 + 16;
      v5 += 32;
      *(_QWORD *)(v11 + 16) = v5;
    }
    if ( v55 == v10 )
    {
      if ( v56 )
      {
        for ( j = (_QWORD *)(v56 + 24LL * *(_QWORD *)(v56 - 8)); (_QWORD *)v56 != j; sub_91D830(j) )
          j -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v55);
    }
    if ( v48 == v10 )
    {
      if ( v49 )
      {
        for ( k = (_QWORD *)(v49 + 24LL * *(_QWORD *)(v49 - 8)); (_QWORD *)v49 != k; sub_91D830(k) )
          k -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v48);
    }
    *(_BYTE *)(a1 + 752) = 1;
LABEL_26:
    v6 += 8;
  }
  while ( v46 != v6 );
  LOBYTE(v4) = v53;
LABEL_69:
  if ( (v4 & 1) != 0 )
  {
    v40 = sub_BD5C60((__int64)a2);
    v62 = &v70;
    v61 = v40;
    v66 = 512;
    v56 = 0x200000000LL;
    v55 = (unsigned int *)v57;
    v70 = &unk_49DA100;
    v63 = &v71;
    v64 = 0;
    v65 = 0;
    v67 = 7;
    v68 = 0;
    v69 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v71 = &unk_49DA0B0;
    sub_D5F1F0((__int64)&v55, (__int64)a2);
    v51 = 257;
    BYTE4(v47) = 1;
    LODWORD(v47) = sub_B45210((__int64)a2);
    if ( v41 == 45 )
      v29 = (_BYTE *)sub_92A220(&v55, a4, a3, v47, (__int64)&v48, 0);
    else
      v29 = (_BYTE *)sub_94AB40(&v55, a4, a3, v47, (__int64)&v48, 0);
    sub_BD84D0((__int64)a2, (__int64)v29);
    sub_D68D20((__int64)&v48, 0, (__int64)a2);
    sub_28F19A0(a1 + 64, &v48);
    sub_D68D70(&v48);
    if ( *v29 <= 0x1Cu )
      v29 = 0;
    nullsub_61();
    v70 = &unk_49DA100;
    nullsub_63();
    if ( v55 != (unsigned int *)v57 )
      _libc_free((unsigned __int64)v55);
    v28 = v52;
  }
  else
  {
    v28 = v52;
    v29 = a2;
  }
LABEL_71:
  if ( v28 != v54 )
    _libc_free((unsigned __int64)v28);
  return v29;
}
