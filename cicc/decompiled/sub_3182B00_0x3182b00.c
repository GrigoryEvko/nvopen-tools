// Function: sub_3182B00
// Address: 0x3182b00
//
_QWORD *__fastcall sub_3182B00(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v7; // r15
  __int16 v8; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r12
  unsigned int *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // r12
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned int *v24; // rax
  __int64 **v25; // r12
  __int64 (__fastcall *v26)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v27; // r15
  __int64 v28; // rdi
  _QWORD *v29; // rax
  unsigned int v30; // esi
  _QWORD *v31; // r12
  __int64 v32; // r8
  int v33; // r11d
  _QWORD *v34; // rdx
  unsigned int v35; // r14d
  unsigned int v36; // edi
  _QWORD *v37; // rax
  _QWORD *v38; // rcx
  unsigned __int64 *v39; // rax
  __int64 v41; // r12
  int v42; // eax
  unsigned int v43; // edx
  unsigned __int8 v44; // cl
  int v45; // r12d
  __int64 v46; // r12
  unsigned int *v47; // r12
  unsigned __int64 v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  int v51; // eax
  int v52; // eax
  int v53; // esi
  int v54; // esi
  __int64 v55; // r8
  unsigned int v56; // ecx
  __int64 v57; // rdi
  int v58; // r11d
  _QWORD *v59; // r9
  int v60; // ecx
  int v61; // ecx
  __int64 v62; // rdi
  _QWORD *v63; // r8
  unsigned int v64; // r14d
  int v65; // r10d
  __int64 v66; // rsi
  unsigned __int64 v67; // [rsp+10h] [rbp-180h]
  __int64 v69; // [rsp+40h] [rbp-150h]
  __int64 v71; // [rsp+68h] [rbp-128h] BYREF
  char v72[32]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v73; // [rsp+90h] [rbp-100h]
  _QWORD v74[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v75; // [rsp+C0h] [rbp-D0h]
  unsigned int *v76; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-B8h]
  _BYTE v78[32]; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+100h] [rbp-90h]
  unsigned __int64 v80; // [rsp+108h] [rbp-88h]
  __int64 v81; // [rsp+110h] [rbp-80h]
  __int64 v82; // [rsp+118h] [rbp-78h]
  void **v83; // [rsp+120h] [rbp-70h]
  void **v84; // [rsp+128h] [rbp-68h]
  __int64 v85; // [rsp+130h] [rbp-60h]
  int v86; // [rsp+138h] [rbp-58h]
  __int16 v87; // [rsp+13Ch] [rbp-54h]
  char v88; // [rsp+13Eh] [rbp-52h]
  __int64 v89; // [rsp+140h] [rbp-50h]
  __int64 v90; // [rsp+148h] [rbp-48h]
  void *v91; // [rsp+150h] [rbp-40h] BYREF
  void *v92; // [rsp+158h] [rbp-38h] BYREF

  v69 = a2;
  if ( !a2 )
    BUG();
  v7 = *(_QWORD *)(a2 + 16);
  v8 = a3;
  v10 = sub_AA48A0(v7);
  v88 = 7;
  v82 = v10;
  v11 = a2;
  v83 = &v91;
  v84 = &v92;
  v76 = (unsigned int *)v78;
  v91 = &unk_49DA100;
  v77 = 0x200000000LL;
  v12 = (__int64)v74;
  v85 = 0;
  v92 = &unk_49DA0B0;
  v86 = 0;
  LOWORD(v81) = v8;
  v87 = 512;
  v89 = 0;
  v90 = 0;
  v79 = v7;
  v80 = a2;
  if ( a2 != v7 + 48 )
  {
    v13 = *(_QWORD *)sub_B46C60(a2 - 24);
    v74[0] = v13;
    if ( v13 && (sub_B96E90((__int64)v74, v13, 1), (v16 = v74[0]) != 0) )
    {
      a2 = (unsigned int)v77;
      v17 = v76;
      v12 = (unsigned int)v77;
      v11 = (__int64)&v76[4 * (unsigned int)v77];
      if ( v76 != (unsigned int *)v11 )
      {
        while ( 1 )
        {
          v14 = *v17;
          if ( !(_DWORD)v14 )
            break;
          v17 += 4;
          if ( (unsigned int *)v11 == v17 )
            goto LABEL_50;
        }
        *((_QWORD *)v17 + 1) = v74[0];
LABEL_10:
        a2 = v16;
        sub_B91220((__int64)v74, v16);
        goto LABEL_11;
      }
LABEL_50:
      if ( (unsigned int)v77 >= (unsigned __int64)HIDWORD(v77) )
      {
        a2 = (unsigned int)v77 + 1LL;
        if ( HIDWORD(v77) < a2 )
        {
          a2 = (unsigned __int64)v78;
          sub_C8D5F0((__int64)&v76, v78, (unsigned int)v77 + 1LL, 0x10u, v14, v15);
          v11 = (__int64)&v76[4 * (unsigned int)v77];
        }
        *(_QWORD *)v11 = 0;
        *(_QWORD *)(v11 + 8) = v16;
        v16 = v74[0];
        LODWORD(v77) = v77 + 1;
      }
      else
      {
        if ( v11 )
        {
          *(_DWORD *)v11 = 0;
          *(_QWORD *)(v11 + 8) = v16;
          LODWORD(v12) = v77;
          v16 = v74[0];
        }
        v12 = (unsigned int)(v12 + 1);
        LODWORD(v77) = v12;
      }
    }
    else
    {
      a2 = 0;
      sub_93FB40((__int64)&v76, 0);
      v16 = v74[0];
    }
    if ( !v16 )
      goto LABEL_11;
    goto LABEL_10;
  }
LABEL_11:
  if ( *(char *)(a4 + 7) < 0 )
  {
    v18 = sub_BD2BC0(a4);
    v19 = v18 + v11;
    if ( *(char *)(a4 + 7) < 0 )
      v19 -= sub_BD2BC0(a4);
    v20 = v19 >> 4;
    if ( (_DWORD)v20 )
    {
      v21 = 0;
      v22 = 16LL * (unsigned int)v20;
      while ( 1 )
      {
        v23 = 0;
        if ( *(char *)(a4 + 7) < 0 )
          v23 = sub_BD2BC0(a4);
        v24 = (unsigned int *)(v21 + v23);
        v11 = *(_QWORD *)v24;
        if ( *(_DWORD *)(*(_QWORD *)v24 + 8LL) == 6 )
          break;
        v21 += 16;
        if ( v22 == v21 )
          goto LABEL_21;
      }
      v11 = *(_DWORD *)(a4 + 4) & 0x7FFFFFF;
      v5 = *(_QWORD *)(a4 + 32 * (v24[2] - v11));
    }
  }
LABEL_21:
  if ( (*(_BYTE *)(v5 + 2) & 1) != 0 )
    sub_B2C6D0(v5, a2, v11, v12);
  v25 = *(__int64 ***)(*(_QWORD *)(v5 + 96) + 8LL);
  v73 = 257;
  if ( v25 == *(__int64 ***)(a4 + 8) )
  {
    v27 = a4;
    goto LABEL_29;
  }
  v26 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v83 + 15);
  if ( v26 != sub_920130 )
  {
    v27 = v26((__int64)v83, 49u, (_BYTE *)a4, (__int64)v25);
    goto LABEL_28;
  }
  if ( *(_BYTE *)a4 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x31u) )
      v27 = sub_ADAB70(49, a4, v25, 0);
    else
      v27 = sub_AA93C0(0x31u, a4, (__int64)v25);
LABEL_28:
    if ( v27 )
      goto LABEL_29;
  }
  v75 = 257;
  v27 = sub_B51D30(49, a4, (__int64)v25, (__int64)v74, 0, 0);
  if ( *(_BYTE *)v27 > 0x1Cu )
  {
    switch ( *(_BYTE *)v27 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_43;
      case 'T':
      case 'U':
      case 'V':
        v41 = *(_QWORD *)(v27 + 8);
        v42 = *(unsigned __int8 *)(v41 + 8);
        v43 = v42 - 17;
        v44 = *(_BYTE *)(v41 + 8);
        if ( (unsigned int)(v42 - 17) <= 1 )
          v44 = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
        if ( v44 <= 3u || v44 == 5 || (v44 & 0xFD) == 4 )
          goto LABEL_43;
        if ( (_BYTE)v42 == 15 )
        {
          if ( (*(_BYTE *)(v41 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v27 + 8)) )
            break;
          v41 = **(_QWORD **)(v41 + 16);
          v42 = *(unsigned __int8 *)(v41 + 8);
          v43 = v42 - 17;
        }
        else if ( (_BYTE)v42 == 16 )
        {
          do
          {
            v41 = *(_QWORD *)(v41 + 24);
            LOBYTE(v42) = *(_BYTE *)(v41 + 8);
          }
          while ( (_BYTE)v42 == 16 );
          v43 = (unsigned __int8)v42 - 17;
        }
        if ( v43 <= 1 )
          LOBYTE(v42) = *(_BYTE *)(**(_QWORD **)(v41 + 16) + 8LL);
        if ( (unsigned __int8)v42 <= 3u || (_BYTE)v42 == 5 || (v42 & 0xFD) == 4 )
        {
LABEL_43:
          v45 = v86;
          if ( v85 )
            sub_B99FD0(v27, 3u, v85);
          sub_B45150(v27, v45);
        }
        break;
      default:
        break;
    }
  }
  (*((void (__fastcall **)(void **, __int64, char *, unsigned __int64, __int64))*v84 + 2))(v84, v27, v72, v80, v81);
  v46 = 4LL * (unsigned int)v77;
  if ( v76 != &v76[v46] )
  {
    v67 = a4;
    v47 = &v76[v46];
    v48 = (unsigned __int64)v76;
    do
    {
      v49 = *(_QWORD *)(v48 + 8);
      v50 = *(_DWORD *)v48;
      v48 += 16LL;
      sub_B99FD0(v27, v50, v49);
    }
    while ( v47 != (unsigned int *)v48 );
    a4 = v67;
  }
LABEL_29:
  v75 = 257;
  v28 = *(_QWORD *)(v5 + 24);
  v71 = v27;
  v29 = sub_31822F0(v28, v5, &v71, 1, (__int64)v74, a5, v69, a3);
  v30 = *(_DWORD *)(a1 + 24);
  v31 = v29;
  if ( !v30 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_83;
  }
  v32 = *(_QWORD *)(a1 + 8);
  v33 = 1;
  v34 = 0;
  v35 = ((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4);
  v36 = (v30 - 1) & v35;
  v37 = (_QWORD *)(v32 + 16LL * v36);
  v38 = (_QWORD *)*v37;
  if ( v31 == (_QWORD *)*v37 )
  {
LABEL_31:
    v39 = v37 + 1;
    goto LABEL_32;
  }
  while ( v38 != (_QWORD *)-4096LL )
  {
    if ( v38 == (_QWORD *)-8192LL && !v34 )
      v34 = v37;
    v36 = (v30 - 1) & (v33 + v36);
    v37 = (_QWORD *)(v32 + 16LL * v36);
    v38 = (_QWORD *)*v37;
    if ( v31 == (_QWORD *)*v37 )
      goto LABEL_31;
    ++v33;
  }
  if ( !v34 )
    v34 = v37;
  v51 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v52 = v51 + 1;
  if ( 4 * v52 >= 3 * v30 )
  {
LABEL_83:
    sub_3182920(a1, 2 * v30);
    v53 = *(_DWORD *)(a1 + 24);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(a1 + 8);
      v56 = v54 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v52 = *(_DWORD *)(a1 + 16) + 1;
      v34 = (_QWORD *)(v55 + 16LL * v56);
      v57 = *v34;
      if ( v31 != (_QWORD *)*v34 )
      {
        v58 = 1;
        v59 = 0;
        while ( v57 != -4096 )
        {
          if ( !v59 && v57 == -8192 )
            v59 = v34;
          v56 = v54 & (v58 + v56);
          v34 = (_QWORD *)(v55 + 16LL * v56);
          v57 = *v34;
          if ( v31 == (_QWORD *)*v34 )
            goto LABEL_77;
          ++v58;
        }
        if ( v59 )
          v34 = v59;
      }
      goto LABEL_77;
    }
    goto LABEL_113;
  }
  if ( v30 - *(_DWORD *)(a1 + 20) - v52 <= v30 >> 3 )
  {
    sub_3182920(a1, v30);
    v60 = *(_DWORD *)(a1 + 24);
    if ( v60 )
    {
      v61 = v60 - 1;
      v62 = *(_QWORD *)(a1 + 8);
      v63 = 0;
      v64 = v61 & v35;
      v65 = 1;
      v52 = *(_DWORD *)(a1 + 16) + 1;
      v34 = (_QWORD *)(v62 + 16LL * v64);
      v66 = *v34;
      if ( v31 != (_QWORD *)*v34 )
      {
        while ( v66 != -4096 )
        {
          if ( !v63 && v66 == -8192 )
            v63 = v34;
          v64 = v61 & (v65 + v64);
          v34 = (_QWORD *)(v62 + 16LL * v64);
          v66 = *v34;
          if ( v31 == (_QWORD *)*v34 )
            goto LABEL_77;
          ++v65;
        }
        if ( v63 )
          v34 = v63;
      }
      goto LABEL_77;
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_77:
  *(_DWORD *)(a1 + 16) = v52;
  if ( *v34 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v34 = v31;
  v39 = v34 + 1;
  v34[1] = 0;
LABEL_32:
  *v39 = a4;
  nullsub_61();
  v91 = &unk_49DA100;
  nullsub_63();
  if ( v76 != (unsigned int *)v78 )
    _libc_free((unsigned __int64)v76);
  return v31;
}
