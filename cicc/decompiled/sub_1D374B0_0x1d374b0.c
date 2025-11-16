// Function: sub_1D374B0
// Address: 0x1d374b0
//
__int64 *__fastcall sub_1D374B0(
        __int64 a1,
        _BYTE *a2,
        const void **a3,
        unsigned int *a4,
        __int64 a5,
        __int64 *a6,
        __m128 a7,
        double a8,
        __m128i a9)
{
  unsigned int *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rdx
  unsigned int *v12; // r13
  __int64 v13; // r8
  __int64 v14; // rdx
  char *v15; // rax
  char *v16; // rdx
  __int64 v17; // r9
  char v18; // al
  const void **v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  char v22; // cl
  __int16 v23; // ax
  const __m128i *v24; // r12
  __int64 v25; // rcx
  __int64 v26; // r8
  const __m128i *v27; // r14
  unsigned __int64 v28; // r15
  __m128 *v29; // rdx
  __int64 v30; // r13
  unsigned int *v31; // rbx
  char v32; // di
  unsigned int *v33; // r12
  __int64 v34; // rax
  char v35; // r15
  const void **v36; // r13
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned int v41; // r14d
  unsigned int v42; // eax
  char v43; // r13
  const void **v44; // rax
  __int64 *v45; // r14
  __int64 v46; // r15
  __int64 *result; // rax
  _QWORD *v48; // r12
  __int64 v49; // rdx
  _QWORD *v50; // r15
  unsigned int v51; // r11d
  __int64 v52; // rdx
  __int64 v53; // r12
  int v54; // ecx
  const void **v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned int v59; // ebx
  __int64 v60; // r8
  __int64 v61; // r9
  unsigned int v62; // eax
  unsigned int v63; // ebx
  __int64 v64; // rsi
  int v65; // edx
  int v66; // eax
  __int64 v67; // rdi
  __int64 (*v68)(); // r9
  __int64 v69; // rax
  int v70; // edx
  __int128 v71; // [rsp-10h] [rbp-1E0h]
  unsigned int v72; // [rsp+0h] [rbp-1D0h]
  unsigned int v73; // [rsp+8h] [rbp-1C8h]
  unsigned __int8 v74; // [rsp+Fh] [rbp-1C1h]
  __int64 v75; // [rsp+10h] [rbp-1C0h]
  unsigned int v76; // [rsp+10h] [rbp-1C0h]
  unsigned int v77; // [rsp+10h] [rbp-1C0h]
  __int64 *v78; // [rsp+20h] [rbp-1B0h]
  __int64 *v80; // [rsp+30h] [rbp-1A0h]
  __int64 *v81; // [rsp+38h] [rbp-198h]
  _BYTE *v82; // [rsp+50h] [rbp-180h] BYREF
  const void **v83; // [rsp+58h] [rbp-178h]
  __int64 v84; // [rsp+60h] [rbp-170h] BYREF
  const void **v85; // [rsp+68h] [rbp-168h]
  char v86[8]; // [rsp+70h] [rbp-160h] BYREF
  __int64 v87; // [rsp+78h] [rbp-158h]
  __int64 v88; // [rsp+80h] [rbp-150h] BYREF
  const void **v89; // [rsp+88h] [rbp-148h]
  const void **v90; // [rsp+90h] [rbp-140h] BYREF
  __int64 v91; // [rsp+98h] [rbp-138h]
  _BYTE v92[304]; // [rsp+A0h] [rbp-130h] BYREF

  v9 = a4;
  v82 = a2;
  v83 = a3;
  if ( a5 == 1 )
    return *(__int64 **)a4;
  v10 = a6;
  v11 = 16 * a5;
  v12 = &a4[4 * a5];
  v13 = (16 * a5) >> 6;
  v14 = v11 >> 4;
  if ( v13 <= 0 )
  {
    v15 = (char *)a4;
LABEL_48:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_51;
        goto LABEL_85;
      }
      if ( *(_WORD *)(*(_QWORD *)v15 + 24LL) != 48 )
      {
LABEL_9:
        if ( v12 != (unsigned int *)v15 )
          goto LABEL_10;
        goto LABEL_51;
      }
      v15 += 16;
    }
    if ( *(_WORD *)(*(_QWORD *)v15 + 24LL) == 48 )
    {
      v15 += 16;
LABEL_85:
      if ( *(_WORD *)(*(_QWORD *)v15 + 24LL) == 48 )
        goto LABEL_51;
      goto LABEL_9;
    }
    goto LABEL_9;
  }
  v13 <<= 6;
  v15 = (char *)a4;
  v16 = (char *)a4 + v13;
  while ( 1 )
  {
    a4 = *(unsigned int **)v15;
    if ( *(_WORD *)(*(_QWORD *)v15 + 24LL) != 48 )
      goto LABEL_9;
    a4 = (unsigned int *)*((_QWORD *)v15 + 2);
    if ( *((_WORD *)a4 + 12) != 48 )
    {
      if ( v12 != (unsigned int *)(v15 + 16) )
        goto LABEL_10;
      goto LABEL_51;
    }
    a4 = (unsigned int *)*((_QWORD *)v15 + 4);
    if ( *((_WORD *)a4 + 12) != 48 )
    {
      if ( v12 != (unsigned int *)(v15 + 32) )
        goto LABEL_10;
LABEL_51:
      v90 = 0;
      LODWORD(v91) = 0;
      v48 = sub_1D2B300(a6, 0x30u, (__int64)&v90, (unsigned int)v82, (__int64)v83, (__int64)a6);
      if ( v90 )
        sub_161E7C0((__int64)&v90, (__int64)v90);
      return v48;
    }
    a4 = (unsigned int *)*((_QWORD *)v15 + 6);
    if ( *((_WORD *)a4 + 12) != 48 )
      break;
    v15 += 64;
    if ( v16 == v15 )
    {
      v14 = ((char *)v12 - v15) >> 4;
      goto LABEL_48;
    }
  }
  if ( v12 == (unsigned int *)(v15 + 48) )
    goto LABEL_51;
LABEL_10:
  v17 = (unsigned __int8)v82;
  if ( (_BYTE)v82 )
  {
    if ( (unsigned __int8)((_BYTE)v82 - 14) > 0x5Fu )
    {
LABEL_12:
      v19 = v83;
    }
    else
    {
      switch ( (char)v82 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v17 = 3;
          v19 = 0;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v17 = 4;
          v19 = 0;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v17 = 5;
          v19 = 0;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v17 = 6;
          v19 = 0;
          break;
        case 55:
          v17 = 7;
          v19 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v17 = 8;
          v19 = 0;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v17 = 9;
          v19 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v17 = 10;
          v19 = 0;
          break;
        default:
          v17 = 2;
          v19 = 0;
          break;
      }
    }
  }
  else
  {
    v18 = sub_1F58D20(&v82);
    v17 = 0;
    if ( !v18 )
      goto LABEL_12;
    v17 = (unsigned int)sub_1F596B0(&v82);
  }
  LOBYTE(v84) = v17;
  v90 = (const void **)v92;
  v85 = v19;
  v91 = 0x1000000000LL;
  if ( v12 == v9 )
    goto LABEL_39;
  v74 = v17;
  v78 = v10;
  do
  {
    v20 = *(_QWORD *)v9;
    v21 = *(_QWORD *)(*(_QWORD *)v9 + 40LL) + 16LL * v9[2];
    v22 = *(_BYTE *)v21;
    v87 = *(_QWORD *)(v21 + 8);
    v23 = *(_WORD *)(v20 + 24);
    v86[0] = v22;
    if ( v23 == 48 )
    {
      v88 = 0;
      LODWORD(v89) = 0;
      v50 = sub_1D2B300(v78, 0x30u, (__int64)&v88, v84, (__int64)v85, v17);
      v13 = v49;
      if ( v88 )
      {
        v75 = v49;
        sub_161E7C0((__int64)&v88, v88);
        v13 = v75;
      }
      if ( v86[0] )
      {
        v51 = word_42E7700[(unsigned __int8)(v86[0] - 14)];
      }
      else
      {
        v76 = v13;
        v56 = sub_1F58D30(v86);
        v13 = v76;
        v51 = v56;
      }
      v52 = (unsigned int)v91;
      v53 = v51;
      a2 = (_BYTE *)(HIDWORD(v91) - (unsigned __int64)(unsigned int)v91);
      v54 = v91;
      if ( v51 > (unsigned __int64)a2 )
      {
        a2 = v92;
        v73 = v13;
        v77 = v51;
        sub_16CD150((__int64)&v90, v92, v51 + (unsigned __int64)(unsigned int)v91, 16, v13, v17);
        v52 = (unsigned int)v91;
        v13 = v73;
        v51 = v77;
        v54 = v91;
      }
      v55 = &v90[2 * v52];
      if ( v53 )
      {
        do
        {
          if ( v55 )
          {
            *v55 = v50;
            *((_DWORD *)v55 + 2) = v13;
          }
          v55 += 2;
          --v53;
        }
        while ( v53 );
        v54 = v91;
      }
      a4 = (unsigned int *)(v51 + v54);
      LODWORD(v91) = (_DWORD)a4;
    }
    else
    {
      if ( v23 != 104 )
      {
        result = 0;
        goto LABEL_45;
      }
      v24 = *(const __m128i **)(v20 + 32);
      v25 = (unsigned int)v91;
      v26 = 40LL * *(unsigned int *)(v20 + 56);
      v27 = (const __m128i *)((char *)v24 + v26);
      v13 = v26 >> 3;
      v28 = 0xCCCCCCCCCCCCCCCDLL * v13;
      if ( 0xCCCCCCCCCCCCCCCDLL * v13 > HIDWORD(v91) - (unsigned __int64)(unsigned int)v91 )
      {
        a2 = v92;
        sub_16CD150((__int64)&v90, v92, v28 + (unsigned int)v91, 16, v13, v17);
        v25 = (unsigned int)v91;
      }
      v29 = (__m128 *)&v90[2 * v25];
      if ( v24 != v27 )
      {
        do
        {
          if ( v29 )
          {
            a7 = (__m128)_mm_loadu_si128(v24);
            *v29 = a7;
          }
          v24 = (const __m128i *)((char *)v24 + 40);
          ++v29;
        }
        while ( v27 != v24 );
        v25 = (unsigned int)v91;
      }
      a4 = (unsigned int *)(v28 + v25);
      LODWORD(v91) = (_DWORD)a4;
    }
    v9 += 4;
  }
  while ( v12 != v9 );
  v19 = v90;
  v17 = v74;
  v10 = v78;
  v30 = 2LL * (unsigned int)v91;
  v31 = (unsigned int *)&v90[v30];
  if ( v90 == &v90[v30] )
    goto LABEL_39;
  v32 = v74;
  v33 = (unsigned int *)v90;
  while ( 2 )
  {
    a4 = *(unsigned int **)v33;
    v34 = *(_QWORD *)(*(_QWORD *)v33 + 40LL) + 16LL * v33[2];
    v35 = *(_BYTE *)v34;
    v36 = *(const void ***)(v34 + 8);
    LOBYTE(v88) = v35;
    v89 = v36;
    if ( v35 == v32 )
    {
      if ( !v32 && v36 != v85 )
        goto LABEL_72;
LABEL_29:
      v33 += 4;
      if ( v31 == v33 )
        break;
      goto LABEL_30;
    }
    if ( v32 )
    {
      v41 = sub_1D13440(v32);
      goto LABEL_34;
    }
LABEL_72:
    v41 = sub_1F58D40(&v84, a2, v19, a4, v13, v17);
LABEL_34:
    if ( v35 )
      v42 = sub_1D13440(v35);
    else
      v42 = sub_1F58D40(&v88, a2, v37, v38, v39, v40);
    if ( v42 <= v41 )
      goto LABEL_29;
    v33 += 4;
    LOBYTE(v84) = v35;
    v85 = v36;
    if ( v31 != v33 )
    {
LABEL_30:
      v32 = v84;
      continue;
    }
    break;
  }
  v10 = v78;
LABEL_39:
  v43 = (char)v82;
  if ( (_BYTE)v82 )
  {
    if ( (unsigned __int8)((_BYTE)v82 - 14) > 0x5Fu )
      goto LABEL_41;
    switch ( (char)v82 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v43 = 3;
        v44 = 0;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v43 = 4;
        v44 = 0;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v43 = 5;
        v44 = 0;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v43 = 6;
        v44 = 0;
        break;
      case 55:
        v43 = 7;
        v44 = 0;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v43 = 8;
        v44 = 0;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v43 = 9;
        v44 = 0;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v43 = 10;
        v44 = 0;
        break;
      default:
        v43 = 2;
        v44 = 0;
        break;
    }
  }
  else if ( (unsigned __int8)sub_1F58D20(&v82) )
  {
    v43 = sub_1F596B0(&v82);
    v44 = v19;
  }
  else
  {
LABEL_41:
    v44 = v83;
  }
  LOBYTE(v88) = v43;
  v89 = v44;
  v45 = (__int64 *)v90;
  v46 = (unsigned int)v91;
  if ( v43 == (_BYTE)v84 )
  {
    if ( !v43 && v44 != v85 )
    {
LABEL_103:
      v59 = sub_1F58D40(&v84, a2, v19, a4, v13, v17);
      goto LABEL_89;
    }
  }
  else
  {
    if ( !(_BYTE)v84 )
      goto LABEL_103;
    v59 = sub_1D13440(v84);
LABEL_89:
    if ( v43 )
      v62 = sub_1D13440(v43);
    else
      v62 = sub_1F58D40(&v88, a2, v57, v58, v60, v61);
    if ( v62 < v59 && &v45[2 * v46] != v45 )
    {
      v80 = &v45[2 * v46];
      v63 = v72;
      do
      {
        v67 = v10[2];
        v68 = *(__int64 (**)())(*(_QWORD *)v67 + 824LL);
        v69 = *(_QWORD *)(*v45 + 40) + 16LL * *((unsigned int *)v45 + 2);
        if ( v68 == sub_1D12E00
          || (LOBYTE(v63) = *(_BYTE *)v69,
              !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, const void **))v68)(
                 v67,
                 v63,
                 *(_QWORD *)(v69 + 8),
                 (unsigned int)v84,
                 v85)) )
        {
          v64 = sub_1D322C0(
                  v10,
                  *v45,
                  v45[1],
                  a1,
                  (unsigned int)v84,
                  v85,
                  *(double *)a7.m128_u64,
                  a8,
                  *(double *)a9.m128i_i64);
          v66 = v65;
        }
        else
        {
          v64 = sub_1D323C0(
                  v10,
                  *v45,
                  v45[1],
                  a1,
                  (unsigned int)v84,
                  v85,
                  *(double *)a7.m128_u64,
                  a8,
                  *(double *)a9.m128i_i64);
          v66 = v70;
        }
        v45 += 2;
        *(v45 - 2) = v64;
        *((_DWORD *)v45 - 2) = v66;
      }
      while ( v80 != v45 );
      v45 = (__int64 *)v90;
      v46 = (unsigned int)v91;
    }
  }
  *((_QWORD *)&v71 + 1) = v46;
  *(_QWORD *)&v71 = v45;
  result = sub_1D359D0(v10, 104, a1, (__int64)v82, v83, 0, *(double *)a7.m128_u64, a8, a9, v71);
LABEL_45:
  if ( v90 != (const void **)v92 )
  {
    v81 = result;
    _libc_free((unsigned __int64)v90);
    return v81;
  }
  return result;
}
