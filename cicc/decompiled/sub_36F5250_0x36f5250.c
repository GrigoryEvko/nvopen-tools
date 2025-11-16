// Function: sub_36F5250
// Address: 0x36f5250
//
__int64 __fastcall sub_36F5250(__int64 a1, __int64 *a2)
{
  __int64 v2; // r15
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 (*v8)(); // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rcx
  int v12; // edi
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r13
  int v18; // eax
  __int64 *v19; // r8
  __int64 v20; // rdx
  unsigned __int8 v21; // r8
  unsigned __int8 v22; // r11
  int v23; // r10d
  __int64 v24; // r13
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r8
  unsigned __int8 v28; // r9
  char v29; // r11
  __int64 v30; // r13
  unsigned __int8 v31; // cl
  __int64 v32; // rax
  unsigned __int8 v33; // cl
  __int64 v34; // r13
  int v35; // eax
  unsigned __int8 v36; // al
  __int64 (*v37)(); // rax
  __int64 (__fastcall *v38)(__int64, __int64 *); // rax
  unsigned __int8 v39; // cl
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // r12
  __int64 v48; // rcx
  __int64 *v49; // rdx
  __int64 v50; // rbx
  unsigned __int64 v51; // rdx
  __int64 *v52; // rdi
  __int64 v53; // rax
  int v54; // edx
  __int64 v55; // rax
  _QWORD *v57; // rax
  char v58; // al
  __int64 *v59; // [rsp+8h] [rbp-C8h]
  __int64 *v60; // [rsp+10h] [rbp-C0h]
  __int64 v62; // [rsp+30h] [rbp-A0h]
  __int64 v63; // [rsp+38h] [rbp-98h]
  _QWORD *v64; // [rsp+38h] [rbp-98h]
  __int64 v65; // [rsp+40h] [rbp-90h]
  unsigned __int8 v66; // [rsp+4Bh] [rbp-85h]
  unsigned __int8 v67; // [rsp+4Ch] [rbp-84h]
  int v68; // [rsp+4Ch] [rbp-84h]
  int v69; // [rsp+5Ch] [rbp-74h] BYREF
  __int64 v70[2]; // [rsp+60h] [rbp-70h] BYREF
  _BYTE *v71; // [rsp+70h] [rbp-60h] BYREF
  __int64 v72; // [rsp+78h] [rbp-58h]
  _BYTE v73[80]; // [rsp+80h] [rbp-50h] BYREF

  v2 = 0;
  v3 = a2[2];
  v4 = *(_QWORD *)v3;
  v5 = *(__int64 (**)())(*(_QWORD *)v3 + 136LL);
  if ( v5 != sub_2DD19D0 )
  {
    v2 = ((__int64 (__fastcall *)(__int64))v5)(v3);
    v4 = *(_QWORD *)v3;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(v4 + 200))(v3);
  v7 = a2[2];
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 136LL);
  if ( v8 == sub_2DD19D0 )
  {
    (*(void (**)(void))(*(_QWORD *)v7 + 200LL))();
    BUG();
  }
  v9 = v8();
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2[2] + 200LL))(a2[2]);
  v12 = *(_DWORD *)(v9 + 8);
  v13 = a2[6];
  v62 = v10;
  v14 = *(_DWORD *)(v9 + 16);
  v15 = *(_QWORD *)(v13 + 8);
  v16 = (unsigned int)-v14;
  if ( v12 == 1 )
    v14 = -v14;
  v17 = v14;
  v18 = *(_DWORD *)(v13 + 32);
  v65 = v17;
  if ( v18 )
  {
    LODWORD(v11) = 0;
    do
    {
      while ( 1 )
      {
        v19 = (__int64 *)(v15 + 40LL * (unsigned int)v11);
        v20 = *v19;
        if ( v12 != 1 )
          break;
        v16 = -v20;
        if ( v17 < v16 )
          v17 = v16;
        v11 = (unsigned int)(v11 + 1);
        if ( v18 == (_DWORD)v11 )
          goto LABEL_15;
      }
      v16 = v19[1] + v20;
      if ( v17 < v16 )
        v17 = v16;
      v11 = (unsigned int)(v11 + 1);
    }
    while ( v18 != (_DWORD)v11 );
  }
LABEL_15:
  v21 = *(_BYTE *)(v13 + 64);
  v67 = v21;
  if ( *(_BYTE *)(v13 + 665) )
  {
    v22 = *(_BYTE *)(v13 + 664);
    v23 = *(_DWORD *)(v13 + 136);
    v11 = v22;
    v16 = -(1LL << v22);
    v24 = v16 & ((1LL << v22) + v17 - 1);
    if ( v23 )
    {
      v25 = -v24;
      if ( v12 != 1 )
        v25 = v24;
      LODWORD(v11) = 0;
      while ( 1 )
      {
        v26 = (int)v11;
        v11 = (unsigned int)(v11 + 1);
        v16 = *(_QWORD *)(v13 + 128) + 16 * v26;
        *(_QWORD *)(v15 + 40LL * (unsigned int)(*(_DWORD *)v16 + v18)) = v25 + *(_QWORD *)(v16 + 8);
        if ( v23 == (_DWORD)v11 )
          break;
        v18 = *(_DWORD *)(v13 + 32);
        v15 = *(_QWORD *)(v13 + 8);
      }
      v15 = *(_QWORD *)(v13 + 8);
      v18 = *(_DWORD *)(v13 + 32);
    }
    v17 = *(_QWORD *)(v13 + 656) + v24;
    if ( v21 > v22 )
      v22 = v21;
    v67 = v22;
  }
  v27 = -858993459 * (unsigned int)((*(_QWORD *)(v13 + 16) - v15) >> 3) - v18;
  if ( (_DWORD)v27 )
  {
    v28 = v67;
    v29 = 0;
    LODWORD(v16) = 0;
    while ( 1 )
    {
      v32 = v15 + 40LL * (unsigned int)(v16 + v18);
      if ( *(_BYTE *)(v32 + 32) && *(_BYTE *)(v13 + 665) )
        goto LABEL_31;
      v11 = *(_QWORD *)(v32 + 8);
      if ( v11 == -1 )
        goto LABEL_31;
      if ( v12 == 1 )
        break;
      v33 = *(_BYTE *)(v32 + 16);
      if ( v28 < v33 )
      {
        v28 = *(_BYTE *)(v32 + 16);
        v29 = 1;
        v15 = -(1LL << v33);
        v34 = v15 & (v17 + (1LL << v33) - 1);
      }
      else
      {
        v15 = 1LL << v33;
        v34 = -(1LL << v33) & (v17 + (1LL << v33) - 1);
      }
      *(_QWORD *)v32 = v34;
      v35 = v16;
      v11 = *(_QWORD *)(v13 + 8);
      v16 = (unsigned int)(v16 + 1);
      v17 = *(_QWORD *)(v11 + 40LL * (unsigned int)(*(_DWORD *)(v13 + 32) + v35) + 8) + v34;
      if ( (_DWORD)v27 == (_DWORD)v16 )
      {
LABEL_40:
        v36 = v67;
        if ( v29 )
          v36 = v28;
        v67 = v36;
        goto LABEL_43;
      }
LABEL_32:
      v18 = *(_DWORD *)(v13 + 32);
      v15 = *(_QWORD *)(v13 + 8);
    }
    v30 = v11 + v17;
    v31 = *(_BYTE *)(v32 + 16);
    if ( v28 < v31 )
    {
      v28 = *(_BYTE *)(v32 + 16);
      v29 = 1;
      v15 = -(1LL << v31);
      v17 = v15 & ((1LL << v31) + v30 - 1);
    }
    else
    {
      v15 = 1LL << v31;
      v17 = -(1LL << v31) & ((1LL << v31) + v30 - 1);
    }
    v11 = -v17;
    *(_QWORD *)v32 = -v17;
LABEL_31:
    v16 = (unsigned int)(v16 + 1);
    if ( (_DWORD)v27 == (_DWORD)v16 )
      goto LABEL_40;
    goto LABEL_32;
  }
LABEL_43:
  v37 = *(__int64 (**)())(*(_QWORD *)v9 + 64LL);
  if ( v37 == sub_2FDBB90
    || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))v37)(v9, v15, v16, v11, v27) )
  {
    v63 = v17;
    if ( !*(_BYTE *)(v13 + 65) )
    {
LABEL_91:
      if ( !*(_BYTE *)(v13 + 36)
        && (!(*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64, __int64, __int64))(*(_QWORD *)v62 + 544LL))(
               v62,
               a2,
               v16,
               v11,
               v27)
         || !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v62 + 536LL))(v62, a2)
         || -858993459 * (unsigned int)((__int64)(*(_QWORD *)(v13 + 16) - *(_QWORD *)(v13 + 8)) >> 3) == *(_DWORD *)(v13 + 32)) )
      {
        v39 = *(_BYTE *)(v9 + 13);
        goto LABEL_53;
      }
LABEL_52:
      v39 = *(_BYTE *)(v9 + 12);
LABEL_53:
      if ( v67 >= v39 )
        v39 = v67;
      v17 = -(1LL << v39) & ((1LL << v39) + v63 - 1);
      goto LABEL_56;
    }
    v38 = *(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v9 + 200LL);
    if ( v38 == sub_2E76F30 )
    {
      if ( (unsigned __int8)sub_B2D610(*a2, 20) )
      {
        v16 = *(unsigned __int8 *)(v13 + 65);
LABEL_48:
        v11 = *(_QWORD *)(v13 + 80);
        if ( v11 == -1 )
          v11 = 0;
        v63 = v17 + v11;
LABEL_51:
        if ( (_BYTE)v16 )
          goto LABEL_52;
        goto LABEL_91;
      }
      v58 = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v9 + 392LL))(v9, a2) ^ 1;
    }
    else
    {
      v58 = ((__int64 (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64))v38)(v9, a2, v16, v11, v27);
    }
    v16 = *(unsigned __int8 *)(v13 + 65);
    if ( !v58 )
      goto LABEL_51;
    goto LABEL_48;
  }
LABEL_56:
  *(_QWORD *)(v13 + 48) = v17 - v65;
  v59 = a2 + 40;
  v60 = (__int64 *)a2[41];
  if ( v60 != a2 + 40 )
  {
    v66 = 0;
    while ( 1 )
    {
      v40 = v60[7];
      if ( (__int64 *)v40 != v60 + 6 )
        break;
LABEL_73:
      v60 = (__int64 *)v60[1];
      if ( v59 == v60 )
      {
        v49 = (__int64 *)a2[41];
        goto LABEL_75;
      }
    }
    while ( 1 )
    {
      v68 = *(_DWORD *)(v40 + 40) & 0xFFFFFF;
      if ( v68 )
        break;
LABEL_71:
      if ( (*(_BYTE *)v40 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v40 + 44) & 8) != 0 )
          v40 = *(_QWORD *)(v40 + 8);
      }
      v40 = *(_QWORD *)(v40 + 8);
      if ( v60 + 6 == (__int64 *)v40 )
        goto LABEL_73;
    }
    v41 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v47 = *(_QWORD *)(v40 + 32) + 40 * v41;
        if ( *(_BYTE *)v47 == 5 )
          break;
LABEL_67:
        if ( v68 == (_DWORD)++v41 )
          goto LABEL_71;
      }
      if ( (unsigned __int16)(*(_WORD *)(v40 + 68) - 14) <= 1u )
      {
        v69 = 0;
        v42 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD, int *))(*(_QWORD *)v2 + 224LL))(
                v2,
                a2,
                *(unsigned int *)(v47 + 24),
                &v69);
        v70[1] = v43;
        v70[0] = v42;
        sub_2EAB560((char *)v47, v69, 0, 0, 0, 0, 0, 0);
        v44 = sub_2E891C0(v40);
        if ( *(_WORD *)(v40 + 68) == 14 )
        {
          v57 = (_QWORD *)sub_2E891C0(v40);
          v46 = sub_2FF7570(v6, v57, 0, v70);
        }
        else
        {
          v64 = (_QWORD *)v44;
          v71 = v73;
          v72 = 0x300000000LL;
          (*(void (__fastcall **)(__int64, __int64 *, _BYTE **))(*(_QWORD *)v6 + 592LL))(v6, v70, &v71);
          v45 = *(_QWORD *)(v40 + 32);
          if ( *(_WORD *)(v40 + 68) != 14 )
            v45 += 80;
          v46 = sub_B0DBA0(v64, v71, (unsigned int)v72, -858993459 * (unsigned int)((v47 - v45) >> 3), 0);
          if ( v71 != v73 )
            _libc_free((unsigned __int64)v71);
        }
        *(_QWORD *)(sub_2E891A0(v40) + 24) = v46;
        goto LABEL_67;
      }
      v48 = (unsigned int)v41++;
      (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v6 + 624LL))(v6, v40, 0, v48, 0);
      v66 = 1;
      if ( v68 == (_DWORD)v41 )
        goto LABEL_71;
    }
  }
  v66 = 0;
  v49 = a2 + 40;
LABEL_75:
  (*(void (__fastcall **)(__int64, __int64 *, __int64 *))(*(_QWORD *)v2 + 96LL))(v2, a2, v49);
  v50 = a2[41];
  if ( v59 != (__int64 *)v50 )
  {
    while ( 1 )
    {
      v51 = *(_QWORD *)(v50 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v52 = (__int64 *)v51;
      if ( v51 != v50 + 48 )
        break;
LABEL_87:
      v50 = *(_QWORD *)(v50 + 8);
      if ( v59 == (__int64 *)v50 )
        return v66;
    }
    if ( !v51 )
      BUG();
    v53 = *(_QWORD *)v51;
    v54 = *(_DWORD *)(v51 + 44);
    if ( (v53 & 4) != 0 )
    {
      if ( (v54 & 4) != 0 )
        goto LABEL_101;
    }
    else if ( (v54 & 4) != 0 )
    {
      while ( 1 )
      {
        v52 = (__int64 *)(v53 & 0xFFFFFFFFFFFFFFF8LL);
        LOBYTE(v54) = *(_DWORD *)((v53 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v54 & 4) == 0 )
          break;
        v53 = *v52;
      }
    }
    if ( (v54 & 8) != 0 )
    {
      LOBYTE(v55) = sub_2E88A90((__int64)v52, 32, 1);
      goto LABEL_85;
    }
LABEL_101:
    v55 = (*(_QWORD *)(v52[2] + 24) >> 5) & 1LL;
LABEL_85:
    if ( (_BYTE)v55 )
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v2 + 104LL))(v2, a2, v50);
    goto LABEL_87;
  }
  return v66;
}
