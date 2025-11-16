// Function: sub_1D42590
// Address: 0x1d42590
//
_QWORD *__fastcall sub_1D42590(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        const void **a4,
        __int64 *a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 v11; // r8
  unsigned __int64 v12; // r13
  const void **v13; // rdx
  unsigned int v14; // r10d
  int v15; // eax
  char v16; // bl
  const void **v17; // rdx
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // edx
  __int64 v24; // rbx
  __int64 v25; // rcx
  __int64 v26; // rbx
  unsigned __int8 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r14
  const void **v31; // r12
  const void **v32; // rax
  const void **v33; // rbx
  char v34; // al
  char v35; // al
  char v36; // al
  unsigned __int8 v37; // dl
  const void **v38; // rdx
  unsigned int v39; // edx
  __int64 v40; // rax
  const void **v41; // rsi
  char v42; // al
  unsigned __int8 v43; // r8
  char v44; // al
  char v45; // al
  unsigned __int8 v46; // al
  unsigned int v47; // eax
  unsigned int v48; // esi
  char v49; // al
  const void **v50; // rdx
  __int128 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // rcx
  __int64 v57; // rsi
  __int64 v58; // rax
  __int128 v59; // [rsp-10h] [rbp-C0h]
  const void **v60; // [rsp+8h] [rbp-A8h]
  __int64 v61; // [rsp+10h] [rbp-A0h]
  const void **v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  char v64; // [rsp+18h] [rbp-98h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int64 v66; // [rsp+18h] [rbp-98h]
  __int64 v67; // [rsp+18h] [rbp-98h]
  unsigned int v68; // [rsp+20h] [rbp-90h]
  unsigned int v69; // [rsp+20h] [rbp-90h]
  unsigned __int8 v70; // [rsp+20h] [rbp-90h]
  unsigned __int8 v71; // [rsp+20h] [rbp-90h]
  __int64 *v72; // [rsp+20h] [rbp-90h]
  __int64 v73; // [rsp+20h] [rbp-90h]
  unsigned int v74; // [rsp+28h] [rbp-88h]
  unsigned int v75; // [rsp+28h] [rbp-88h]
  unsigned int v76; // [rsp+28h] [rbp-88h]
  unsigned int v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  __int64 v79; // [rsp+30h] [rbp-80h] BYREF
  const void **v80; // [rsp+38h] [rbp-78h]
  unsigned int v81; // [rsp+40h] [rbp-70h] BYREF
  const void **v82; // [rsp+48h] [rbp-68h]
  __int64 v83; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v84; // [rsp+58h] [rbp-58h]
  __int64 v85; // [rsp+60h] [rbp-50h] BYREF
  const void **v86; // [rsp+68h] [rbp-48h] BYREF
  __int64 v87; // [rsp+70h] [rbp-40h]

  v11 = (unsigned int)a3;
  v12 = a2;
  v79 = a3;
  v80 = a4;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(a3 - 14) <= 0x5Fu )
    {
      switch ( (char)a3 )
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
          LOBYTE(v11) = 3;
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
          LOBYTE(v11) = 4;
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
          LOBYTE(v11) = 5;
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
          LOBYTE(v11) = 6;
          break;
        case 55:
          LOBYTE(v11) = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v11) = 8;
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
          LOBYTE(v11) = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v11) = 10;
          break;
        default:
          LOBYTE(v11) = 2;
          break;
      }
      goto LABEL_25;
    }
    goto LABEL_3;
  }
  v34 = sub_1F58D20(&v79);
  v11 = 0;
  if ( !v34 )
  {
LABEL_3:
    v13 = v80;
    goto LABEL_4;
  }
  v11 = (unsigned int)sub_1F596B0(&v79);
LABEL_4:
  LOBYTE(v85) = v11;
  v86 = v13;
  if ( (_BYTE)v11 )
  {
LABEL_25:
    v14 = sub_1D13440(v11);
    goto LABEL_6;
  }
  v14 = sub_1F58D40(&v85, a2, v13, a4, v11, a6);
LABEL_6:
  v15 = *(unsigned __int16 *)(a1 + 24);
  if ( v15 != 32 && v15 != 10 )
  {
    v16 = v79;
    if ( (_BYTE)v79 )
    {
      if ( (unsigned __int8)(v79 - 14) <= 0x5Fu )
      {
        switch ( (char)v79 )
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
            LOBYTE(v81) = 3;
            v82 = 0;
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
            LOBYTE(v81) = 4;
            v82 = 0;
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
            LOBYTE(v81) = 5;
            v82 = 0;
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
            LOBYTE(v81) = 6;
            v82 = 0;
            break;
          case 55:
            LOBYTE(v81) = 7;
            v82 = 0;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            LOBYTE(v81) = 8;
            v16 = 8;
            v82 = 0;
            goto LABEL_66;
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
            LOBYTE(v81) = 9;
            v16 = 9;
            v82 = 0;
            goto LABEL_66;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            LOBYTE(v81) = 10;
            v16 = 10;
            v82 = 0;
            goto LABEL_66;
          default:
            LOBYTE(v81) = 2;
            v82 = 0;
            break;
        }
        goto LABEL_13;
      }
    }
    else
    {
      v69 = v14;
      v35 = sub_1F58D20(&v79);
      v14 = v69;
      if ( v35 )
      {
        v36 = sub_1F596B0(&v79);
        v14 = v69;
        v16 = v36;
        goto LABEL_11;
      }
    }
    v17 = v80;
LABEL_11:
    LOBYTE(v81) = v16;
    v82 = v17;
    if ( !v16 )
    {
      v74 = v14;
      v18 = sub_1F58CF0(&v81);
      v14 = v74;
      if ( v18 )
        goto LABEL_13;
      v47 = sub_1F58D40(&v81, a2, v19, v20, v21, v22);
      v14 = v74;
      v48 = v47;
LABEL_56:
      if ( v48 == 32 )
      {
        v49 = 5;
      }
      else if ( v48 > 0x20 )
      {
        if ( v48 == 64 )
        {
          v49 = 6;
        }
        else
        {
          if ( v48 != 128 )
          {
LABEL_63:
            v76 = v14;
            v49 = sub_1F58CC0(a5[6]);
            v14 = v76;
            goto LABEL_61;
          }
          v49 = 7;
        }
      }
      else if ( v48 == 8 )
      {
        v49 = 3;
      }
      else
      {
        v49 = 4;
        if ( v48 != 16 )
        {
          v49 = 2;
          if ( v48 != 1 )
            goto LABEL_63;
        }
      }
      v50 = 0;
LABEL_61:
      LOBYTE(v81) = v49;
      v82 = v50;
      goto LABEL_13;
    }
    if ( (unsigned __int8)(v16 - 2) > 5u && (unsigned __int8)(v16 - 14) > 0x47u )
    {
LABEL_66:
      v48 = sub_1D13440(v16);
      goto LABEL_56;
    }
LABEL_13:
    *((_QWORD *)&v59 + 1) = v12;
    *(_QWORD *)&v59 = a1;
    v75 = v14;
    v24 = sub_1D309E0(a5, 143, a6, v81, v82, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v59);
    v25 = v24;
    v68 = v23;
    if ( v75 > 8 )
    {
      LODWORD(v86) = 8;
      v85 = 1;
      sub_16A8BC0((__int64)&v83, v75, (__int64)&v85);
      if ( (unsigned int)v86 > 0x40 && v85 )
        j_j___libc_free_0_0(v85);
      *(_QWORD *)&v51 = sub_1D38970((__int64)a5, (__int64)&v83, a6, v81, v82, 0, a7, a8, a9, 0);
      v12 = v68 | v12 & 0xFFFFFFFF00000000LL;
      v52 = sub_1D332F0(a5, 54, a6, v81, v82, 0, *(double *)a7.m128i_i64, a8, a9, v24, v12, v51);
      v25 = (__int64)v52;
      if ( v84 > 0x40 && v83 )
      {
        v77 = v23;
        v72 = v52;
        j_j___libc_free_0_0(v83);
        v25 = (__int64)v72;
        v23 = v77;
      }
    }
    v26 = v23;
    v27 = v79;
    v28 = *(_QWORD *)(v25 + 40) + 16LL * v23;
    if ( (_BYTE)v79 == *(_BYTE *)v28 )
    {
      if ( (_BYTE)v79 || v80 == *(const void ***)(v28 + 8) )
        return (_QWORD *)v25;
      v62 = *(const void ***)(v28 + 8);
      v65 = v25;
      v70 = v79;
      v44 = sub_1F58CF0(&v79);
      v43 = v70;
      v25 = v65;
      v41 = v62;
      if ( !v44 )
        goto LABEL_52;
    }
    else
    {
      if ( (_BYTE)v79 )
      {
        v37 = v79 - 14;
        if ( (unsigned __int8)(v79 - 2) <= 5u || v37 <= 0x47u )
          return sub_1D35F20(
                   a5,
                   (unsigned int)v79,
                   v80,
                   a6,
                   v25,
                   v26 | v12 & 0xFFFFFFFF00000000LL,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9);
        if ( v37 <= 0x5Fu )
        {
          switch ( (char)v79 )
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
              v27 = 3;
              v38 = 0;
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
              v27 = 4;
              v38 = 0;
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
              v27 = 5;
              v38 = 0;
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
              v27 = 6;
              v38 = 0;
              break;
            case 55:
              v27 = 7;
              v38 = 0;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v27 = 8;
              v38 = 0;
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
              v27 = 9;
              v38 = 0;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v27 = 10;
              v38 = 0;
              break;
            default:
              v27 = 2;
              v38 = 0;
              break;
          }
LABEL_40:
          v12 = v26 | v12 & 0xFFFFFFFF00000000LL;
          v25 = sub_1D32840(a5, v27, v38, v25, v12, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64);
          v26 = v39;
          v40 = *(_QWORD *)(v25 + 40) + 16LL * v39;
          if ( (_BYTE)v79 == *(_BYTE *)v40 )
          {
            if ( (_BYTE)v79 )
              return (_QWORD *)v25;
            v41 = *(const void ***)(v40 + 8);
            goto LABEL_43;
          }
          return sub_1D35F20(
                   a5,
                   (unsigned int)v79,
                   v80,
                   a6,
                   v25,
                   v26 | v12 & 0xFFFFFFFF00000000LL,
                   *(double *)a7.m128i_i64,
                   a8,
                   a9);
        }
LABEL_39:
        v38 = v80;
        goto LABEL_40;
      }
      v60 = *(const void ***)(v28 + 8);
      v61 = v25;
      v64 = *(_BYTE *)v28;
      v42 = sub_1F58CF0(&v79);
      v43 = 0;
      v25 = v61;
      v41 = v60;
      if ( !v42 )
      {
LABEL_52:
        v66 = v25;
        v71 = v43;
        v45 = sub_1F58D20(&v79);
        v27 = v71;
        v25 = v66;
        if ( v45 )
        {
          v46 = sub_1F596B0(&v79);
          v25 = v66;
          v27 = v46;
          goto LABEL_40;
        }
        goto LABEL_39;
      }
      if ( v64 )
        return sub_1D35F20(
                 a5,
                 (unsigned int)v79,
                 v80,
                 a6,
                 v25,
                 v26 | v12 & 0xFFFFFFFF00000000LL,
                 *(double *)a7.m128i_i64,
                 a8,
                 a9);
    }
LABEL_43:
    if ( v41 == v80 )
      return (_QWORD *)v25;
    return sub_1D35F20(
             a5,
             (unsigned int)v79,
             v80,
             a6,
             v25,
             v26 | v12 & 0xFFFFFFFF00000000LL,
             *(double *)a7.m128i_i64,
             a8,
             a9);
  }
  sub_16A8BC0((__int64)&v83, v14, *(_QWORD *)(a1 + 88) + 24LL);
  if ( (_BYTE)v79 )
  {
    if ( (unsigned __int8)(v79 - 14) <= 0x47u || (unsigned __int8)(v79 - 2) <= 5u )
      goto LABEL_20;
LABEL_27:
    v31 = (const void **)sub_1D15FA0((unsigned int)v79, (__int64)v80);
    v32 = (const void **)sub_16982C0();
    v33 = v32;
    if ( v31 == v32 )
      sub_169D060(&v86, (__int64)v32, &v83);
    else
      sub_169D050((__int64)&v86, v31, &v83);
    v29 = (__int64)sub_1D36490((__int64)a5, (__int64)&v85, a6, v79, v80, 0, *(double *)a7.m128i_i64, a8, a9);
    if ( v86 == v33 )
    {
      v67 = v87;
      if ( v87 )
      {
        v53 = v87 + 32LL * *(_QWORD *)(v87 - 8);
        if ( v87 != v53 )
        {
          do
          {
            v53 -= 32;
            if ( v33 == *(const void ***)(v53 + 8) )
            {
              v54 = *(_QWORD *)(v53 + 16);
              v63 = v54;
              if ( v54 )
              {
                v55 = v54 + 32LL * *(_QWORD *)(v54 - 8);
                if ( v54 != v55 )
                {
                  do
                  {
                    v55 -= 32;
                    if ( v33 == *(const void ***)(v55 + 8) )
                    {
                      v56 = *(_QWORD *)(v55 + 16);
                      if ( v56 )
                      {
                        v57 = 32LL * *(_QWORD *)(v56 - 8);
                        v58 = v56 + v57;
                        if ( v56 != v56 + v57 )
                        {
                          do
                          {
                            v73 = v56;
                            v78 = v58 - 32;
                            sub_127D120((_QWORD *)(v58 - 24));
                            v58 = v78;
                            v56 = v73;
                          }
                          while ( v73 != v78 );
                        }
                        j_j_j___libc_free_0_0(v56 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v55 + 8);
                    }
                  }
                  while ( v63 != v55 );
                }
                j_j_j___libc_free_0_0(v63 - 8);
              }
            }
            else
            {
              sub_1698460(v53 + 8);
            }
          }
          while ( v67 != v53 );
        }
        j_j_j___libc_free_0_0(v67 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v86);
    }
    goto LABEL_21;
  }
  if ( !(unsigned __int8)sub_1F58CF0(&v79) )
    goto LABEL_27;
LABEL_20:
  v29 = sub_1D38970((__int64)a5, (__int64)&v83, a6, v79, v80, 0, a7, a8, a9, 0);
LABEL_21:
  if ( v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  return (_QWORD *)v29;
}
