// Function: sub_2785600
// Address: 0x2785600
//
__int64 __fastcall sub_2785600(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int8 *v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r14
  unsigned __int64 v13; // r12
  bool v14; // r15
  __int64 v15; // r14
  int v16; // eax
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // r14
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned __int8 *v22; // r14
  unsigned __int8 v23; // al
  __int64 *v24; // r15
  void *v25; // r12
  unsigned __int8 v26; // al
  int v27; // eax
  const void **v28; // r15
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rax
  int v34; // edx
  int v35; // edx
  int v36; // r8d
  unsigned __int8 v37; // al
  unsigned __int64 v38; // rax
  __int64 v39; // r15
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  int v43; // edx
  const void *v44; // rax
  _QWORD *i; // r14
  unsigned __int64 v46; // r12
  int v47; // r9d
  const void *v48; // rax
  const void *v49; // rax
  bool v50; // cc
  const void *v51; // rax
  __int64 v54; // [rsp+20h] [rbp-140h]
  unsigned __int8 *v55; // [rsp+28h] [rbp-138h]
  char v56; // [rsp+3Fh] [rbp-121h] BYREF
  unsigned __int64 v57; // [rsp+40h] [rbp-120h] BYREF
  unsigned int v58; // [rsp+48h] [rbp-118h]
  char v59; // [rsp+4Ch] [rbp-114h]
  unsigned __int64 v60; // [rsp+50h] [rbp-110h] BYREF
  unsigned int v61; // [rsp+58h] [rbp-108h]
  void *v62; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v63; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v64; // [rsp+70h] [rbp-F0h]
  unsigned int v65; // [rsp+78h] [rbp-E8h]
  const void *v66; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v67; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v68; // [rsp+90h] [rbp-D0h] BYREF
  unsigned int v69; // [rsp+98h] [rbp-C8h]
  const void **v70; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+A8h] [rbp-B8h]
  _BYTE v72[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v70 = (const void **)v72;
  v71 = 0x400000000LL;
  v4 = 32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF);
  if ( (a3[7] & 0x40) != 0 )
  {
    v5 = (unsigned __int8 *)*((_QWORD *)a3 - 1);
    v55 = &v5[v4];
  }
  else
  {
    v55 = a3;
    v5 = &a3[-v4];
  }
  if ( v5 != v55 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = *(unsigned __int8 **)v5;
        v23 = **(_BYTE **)v5;
        if ( v23 > 0x1Cu )
          break;
        if ( v23 != 18 )
          BUG();
        v24 = (__int64 *)(v22 + 24);
        v25 = sub_C33340();
        if ( *((void **)v22 + 3) == v25 )
        {
          v37 = *(_BYTE *)(*((_QWORD *)v22 + 4) + 20LL) & 7;
          if ( v37 <= 1u )
            goto LABEL_65;
          if ( v37 != 3 || (*(_BYTE *)(*((_QWORD *)v22 + 4) + 20LL) & 8) == 0 )
            goto LABEL_84;
        }
        else
        {
          v26 = v22[44] & 7;
          if ( v26 <= 1u )
            goto LABEL_65;
          if ( v26 != 3 || (v22[44] & 8) == 0 )
            goto LABEL_35;
        }
        if ( (unsigned __int8)sub_920620((__int64)a3) && !sub_B451E0((__int64)a3) )
        {
LABEL_65:
          sub_2784ED0((__int64)&v66);
LABEL_66:
          *(_DWORD *)(a1 + 8) = v67;
          *(_QWORD *)a1 = v66;
          *(_DWORD *)(a1 + 24) = v69;
          v38 = v68;
          *(_BYTE *)(a1 + 32) = 1;
          *(_QWORD *)(a1 + 16) = v38;
          v39 = (__int64)v70;
          goto LABEL_67;
        }
        if ( *((void **)v22 + 3) != v25 )
        {
LABEL_35:
          sub_C33EB0(&v62, v24);
          goto LABEL_36;
        }
LABEL_84:
        sub_C3C790(&v62, (_QWORD **)v24);
LABEL_36:
        if ( v25 == v62 )
        {
          if ( (unsigned int)sub_C3E740(&v62, 1u) )
            goto LABEL_95;
        }
        else if ( (unsigned int)sub_C3BAB0((__int64)&v62, 1) )
        {
          goto LABEL_95;
        }
        if ( v25 == v62 )
          v27 = sub_C3E510((__int64)&v62, (__int64)v24);
        else
          v27 = sub_C37950((__int64)&v62, (__int64)v24);
        if ( v27 != 1 )
        {
LABEL_95:
          sub_2784ED0((__int64)&v66);
          *(_DWORD *)(a1 + 8) = v67;
          v44 = v66;
          *(_BYTE *)(a1 + 32) = 1;
          *(_QWORD *)a1 = v44;
          *(_DWORD *)(a1 + 24) = v69;
          *(_QWORD *)(a1 + 16) = v68;
          sub_91D830(&v62);
          v39 = (__int64)v70;
          goto LABEL_67;
        }
        v58 = qword_4FFB348 + 1;
        if ( (unsigned int)(qword_4FFB348 + 1) > 0x40 )
          sub_C43690((__int64)&v57, 0, 0);
        else
          v57 = 0;
        v59 = 0;
        sub_C41980((void **)v24, (__int64)&v57, 1, &v56);
        v61 = v58;
        if ( v58 > 0x40 )
          sub_C43780((__int64)&v60, (const void **)&v57);
        else
          v60 = v57;
        v28 = &v66;
        sub_AADBC0((__int64)&v66, (__int64 *)&v60);
        v29 = (unsigned int)v71;
        v30 = (unsigned int)v71 + 1LL;
        v31 = v71;
        if ( v30 > HIDWORD(v71) )
        {
          if ( v70 > &v66 || (v54 = (__int64)v70, &v66 >= &v70[4 * (unsigned int)v71]) )
          {
            sub_9D5330((__int64)&v70, v30);
            v29 = (unsigned int)v71;
            v32 = (__int64)v70;
            v31 = v71;
          }
          else
          {
            sub_9D5330((__int64)&v70, v30);
            v32 = (__int64)v70;
            v29 = (unsigned int)v71;
            v28 = (const void **)((char *)&v66 + (_QWORD)v70 - v54);
            v31 = v71;
          }
        }
        else
        {
          v32 = (__int64)v70;
        }
        v33 = v32 + 32 * v29;
        if ( v33 )
        {
          v34 = *((_DWORD *)v28 + 2);
          *((_DWORD *)v28 + 2) = 0;
          *(_DWORD *)(v33 + 8) = v34;
          *(_QWORD *)v33 = *v28;
          v35 = *((_DWORD *)v28 + 6);
          *((_DWORD *)v28 + 6) = 0;
          v31 = v71;
          *(_DWORD *)(v33 + 24) = v35;
          *(_QWORD *)(v33 + 16) = v28[2];
        }
        LODWORD(v71) = v31 + 1;
        if ( v69 > 0x40 && v68 )
          j_j___libc_free_0_0(v68);
        if ( v67 > 0x40 && v66 )
          j_j___libc_free_0_0((unsigned __int64)v66);
        if ( v61 > 0x40 && v60 )
          j_j___libc_free_0_0(v60);
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        if ( v25 == v62 )
        {
          if ( v63 )
          {
            for ( i = &v63[3 * *(v63 - 1)]; v63 != i; sub_91D830(i) )
              i -= 3;
            j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
          }
          goto LABEL_28;
        }
        v5 += 32;
        sub_C338F0((__int64)&v62);
        if ( v55 == v5 )
          goto LABEL_63;
      }
      v6 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v6 )
        goto LABEL_90;
      v7 = *(_QWORD *)(a2 + 8);
      v8 = (v6 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v9 = v7 + 16LL * v8;
      v10 = *(unsigned __int8 **)v9;
      if ( v22 != *(unsigned __int8 **)v9 )
        break;
LABEL_7:
      v11 = *(_QWORD *)(a2 + 32);
      if ( v9 != v7 + 16 * v6 )
      {
        v12 = v11 + 40LL * *(unsigned int *)(v9 + 8);
        goto LABEL_9;
      }
LABEL_91:
      v12 = v11 + 40LL * *(unsigned int *)(a2 + 40);
LABEL_9:
      v13 = v12 + 8;
      sub_2784F00((__int64)&v66);
      if ( *(_DWORD *)(v12 + 16) <= 0x40u )
      {
        v14 = 0;
        if ( *(const void **)(v12 + 8) != v66 )
          goto LABEL_13;
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
          goto LABEL_12;
      }
      else
      {
        v14 = sub_C43C50(v12 + 8, &v66);
        if ( !v14 )
          goto LABEL_13;
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
        {
LABEL_12:
          v14 = sub_C43C50(v12 + 24, (const void **)&v68);
          goto LABEL_13;
        }
      }
      v14 = *(_QWORD *)(v12 + 24) == v68;
LABEL_13:
      if ( v69 > 0x40 && v68 )
        j_j___libc_free_0_0(v68);
      if ( v67 > 0x40 && v66 )
        j_j___libc_free_0_0((unsigned __int64)v66);
      v15 = (unsigned int)v71;
      v16 = v71;
      v17 = (__int64)v70;
      if ( v14 )
      {
        v39 = (__int64)v70;
        *(_BYTE *)(a1 + 32) = 0;
        goto LABEL_67;
      }
      v18 = (unsigned int)v71 + 1LL;
      if ( v18 > HIDWORD(v71) )
      {
        if ( (unsigned __int64)v70 > v13 || v13 >= (unsigned __int64)&v70[4 * (unsigned int)v71] )
        {
          sub_9D5330((__int64)&v70, v18);
          v15 = (unsigned int)v71;
          v17 = (__int64)v70;
          v16 = v71;
        }
        else
        {
          v46 = v13 - (_QWORD)v70;
          sub_9D5330((__int64)&v70, v18);
          v17 = (__int64)v70;
          v15 = (unsigned int)v71;
          v13 = (unsigned __int64)v70 + v46;
          v16 = v71;
        }
      }
      v19 = v17 + 32 * v15;
      if ( v19 )
      {
        v20 = *(_DWORD *)(v13 + 8);
        *(_DWORD *)(v19 + 8) = v20;
        if ( v20 > 0x40 )
          sub_C43780(v19, (const void **)v13);
        else
          *(_QWORD *)v19 = *(_QWORD *)v13;
        v21 = *(_DWORD *)(v13 + 24);
        *(_DWORD *)(v19 + 24) = v21;
        if ( v21 > 0x40 )
          sub_C43780(v19 + 16, (const void **)(v13 + 16));
        else
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v13 + 16);
        v16 = v71;
      }
      LODWORD(v71) = v16 + 1;
LABEL_28:
      v5 += 32;
      if ( v55 == v5 )
        goto LABEL_63;
    }
    v43 = 1;
    while ( v10 != (unsigned __int8 *)-4096LL )
    {
      v47 = v43 + 1;
      v8 = (v6 - 1) & (v43 + v8);
      v9 = v7 + 16LL * v8;
      v10 = *(unsigned __int8 **)v9;
      if ( v22 == *(unsigned __int8 **)v9 )
        goto LABEL_7;
      v43 = v47;
    }
LABEL_90:
    v11 = *(_QWORD *)(a2 + 32);
    goto LABEL_91;
  }
LABEL_63:
  v36 = *a3 - 29;
  switch ( *a3 )
  {
    case ')':
      v67 = *((_DWORD *)v70 + 2);
      if ( v67 > 0x40 )
        sub_C43690((__int64)&v66, 0, 0);
      else
        v66 = 0;
      sub_AADBC0((__int64)&v62, (__int64 *)&v66);
      if ( v67 > 0x40 && v66 )
        j_j___libc_free_0_0((unsigned __int64)v66);
      sub_AB51C0((__int64)&v66, (__int64)&v62, (__int64)v70);
      v50 = v65 <= 0x40;
      *(_DWORD *)(a1 + 8) = v67;
      v51 = v66;
      *(_BYTE *)(a1 + 32) = 1;
      *(_QWORD *)a1 = v51;
      *(_DWORD *)(a1 + 24) = v69;
      *(_QWORD *)(a1 + 16) = v68;
      if ( !v50 && v64 )
        j_j___libc_free_0_0(v64);
      if ( (unsigned int)v63 <= 0x40 || !v62 )
        goto LABEL_117;
      j_j___libc_free_0_0((unsigned __int64)v62);
      v39 = (__int64)v70;
      break;
    case '+':
    case '-':
    case '/':
      sub_ABCAA0((__int64)&v66, (__int64)v70, v36, (__int64 *)v70 + 4);
      v39 = (__int64)v70;
      *(_DWORD *)(a1 + 8) = v67;
      v48 = v66;
      *(_BYTE *)(a1 + 32) = 1;
      *(_QWORD *)a1 = v48;
      *(_DWORD *)(a1 + 24) = v69;
      *(_QWORD *)(a1 + 16) = v68;
      break;
    case 'F':
    case 'G':
      sub_AB49F0((__int64)&v66, (__int64)v70, v36, qword_4FFB348 + 1);
      goto LABEL_66;
    case 'S':
      sub_AB3510((__int64)&v66, (__int64)v70, (__int64)(v70 + 4), 0);
      *(_DWORD *)(a1 + 8) = v67;
      v49 = v66;
      *(_BYTE *)(a1 + 32) = 1;
      *(_QWORD *)a1 = v49;
      *(_DWORD *)(a1 + 24) = v69;
      *(_QWORD *)(a1 + 16) = v68;
LABEL_117:
      v39 = (__int64)v70;
      break;
    default:
      BUG();
  }
LABEL_67:
  v40 = v39 + 32LL * (unsigned int)v71;
  if ( v39 != v40 )
  {
    do
    {
      v40 -= 32LL;
      if ( *(_DWORD *)(v40 + 24) > 0x40u )
      {
        v41 = *(_QWORD *)(v40 + 16);
        if ( v41 )
          j_j___libc_free_0_0(v41);
      }
      if ( *(_DWORD *)(v40 + 8) > 0x40u && *(_QWORD *)v40 )
        j_j___libc_free_0_0(*(_QWORD *)v40);
    }
    while ( v39 != v40 );
    v40 = (unsigned __int64)v70;
  }
  if ( (_BYTE *)v40 != v72 )
    _libc_free(v40);
  return a1;
}
