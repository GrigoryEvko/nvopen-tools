// Function: sub_175F2F0
// Address: 0x175f2f0
//
__int64 __fastcall sub_175F2F0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  int v14; // eax
  int v15; // r13d
  __int64 **v16; // rax
  int v17; // r8d
  int v18; // r15d
  void *v19; // rax
  int v20; // r8d
  __int64 *v21; // rsi
  void *v22; // rax
  int v23; // r8d
  __int64 *v24; // rdi
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 v28; // r12
  __int64 v30; // rdi
  int v31; // eax
  void *v32; // rax
  __int64 v33; // r13
  __int64 v34; // r13
  int v35; // ebx
  __int64 *v36; // rdi
  __int64 v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 *v40; // rbx
  __int64 v41; // rax
  double v42; // xmm4_8
  double v43; // xmm5_8
  void *v44; // rax
  __int64 v45; // r13
  __int64 v46; // rsi
  __int64 v47; // rsi
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 **v53; // rsi
  __int64 v54; // rsi
  void **v55; // rdi
  int v56; // r13d
  bool v57; // zf
  __int16 v58; // ax
  __int64 v59; // r15
  _QWORD *v60; // rax
  __int16 v61; // ax
  __int16 v62; // ax
  __int16 v63; // ax
  __int64 v64; // [rsp+0h] [rbp-B0h]
  unsigned int v65; // [rsp+8h] [rbp-A8h]
  int v66; // [rsp+Ch] [rbp-A4h]
  int v67; // [rsp+10h] [rbp-A0h]
  int v68; // [rsp+10h] [rbp-A0h]
  void *v69; // [rsp+10h] [rbp-A0h]
  int v70; // [rsp+10h] [rbp-A0h]
  void *v71; // [rsp+10h] [rbp-A0h]
  __int64 v72; // [rsp+10h] [rbp-A0h]
  int v73; // [rsp+18h] [rbp-98h]
  unsigned int v74; // [rsp+18h] [rbp-98h]
  char v75; // [rsp+1Fh] [rbp-91h]
  __int64 v76; // [rsp+20h] [rbp-90h]
  __int64 v78; // [rsp+30h] [rbp-80h]
  char v80; // [rsp+4Fh] [rbp-61h] BYREF
  unsigned __int64 v81; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v82; // [rsp+58h] [rbp-58h]
  bool v83; // [rsp+5Ch] [rbp-54h]
  char v84[8]; // [rsp+60h] [rbp-50h] BYREF
  void *v85; // [rsp+68h] [rbp-48h] BYREF
  __int64 v86; // [rsp+70h] [rbp-40h]

  v76 = (__int64)(a4 + 3);
  v14 = sub_16431F0(*(_QWORD *)a3);
  if ( v14 != -1 )
  {
    v15 = v14;
    if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
      v16 = *(__int64 ***)(a3 - 8);
    else
      v16 = (__int64 **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    v78 = **v16;
    v75 = *(_BYTE *)(a3 + 16);
    v17 = *(_WORD *)(a2 + 18) & 0x7FF7;
    if ( v17 == 6 || v17 == 1 )
    {
      v80 = 0;
      v82 = *(_DWORD *)(v78 + 8) >> 8;
      if ( v82 <= 0x40 )
      {
        v81 = 0;
      }
      else
      {
        v67 = v17;
        sub_16A4EF0((__int64)&v81, 0, 0);
        v17 = v67;
      }
      v83 = v75 == 65;
      v68 = v17;
      sub_169E1A0(v76, (__int64)&v81, 0, &v80);
      if ( !v80 )
      {
        v19 = sub_16982C0();
        v20 = v68;
        v69 = v19;
        v21 = a4 + 4;
        v73 = v20;
        if ( (void *)a4[4] == v19 )
        {
          sub_169C6E0(&v85, (__int64)v21);
          v23 = v73;
          v22 = v69;
        }
        else
        {
          sub_16986C0(&v85, v21);
          v22 = v69;
          v23 = v73;
        }
        v70 = v23;
        if ( v85 == v22 )
          sub_169EBA0(&v85, 0);
        else
          sub_169D440((__int64)&v85, 0);
        if ( (unsigned int)sub_14A9E40(v76, (__int64)v84) != 1 )
        {
          v24 = *(__int64 **)(a1[1] + 24);
          if ( v70 == 1 )
            v25 = sub_159C540(v24);
          else
            v25 = sub_159C4F0(v24);
          v28 = sub_170E100(a1, a2, v25, a5, a6, a7, a8, v26, v27, a11, a12);
          sub_127D120(&v85);
          sub_135E100((__int64 *)&v81);
          return v28;
        }
        sub_127D120(&v85);
      }
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
    }
    v18 = sub_16431D0(v78);
    if ( v15 >= v18 )
    {
LABEL_7:
      switch ( *(_WORD *)(a2 + 18) & 0x7FFF )
      {
        case 0:
        case 6:
        case 0xE:
          v66 = 33;
          goto LABEL_29;
        case 1:
        case 9:
          v66 = 32;
LABEL_29:
          v74 = sub_16431D0(v78);
          if ( v75 != 65 )
            goto LABEL_30;
          goto LABEL_63;
        case 2:
        case 0xA:
          if ( v75 != 65 )
          {
            v66 = 38;
            goto LABEL_49;
          }
          v66 = 34;
          break;
        case 3:
        case 0xB:
          if ( v75 != 65 )
          {
            v66 = 39;
            goto LABEL_49;
          }
          v66 = 35;
          break;
        case 4:
        case 0xC:
          if ( v75 != 65 )
          {
            v66 = 40;
            goto LABEL_49;
          }
          v66 = 36;
          break;
        case 5:
        case 0xD:
          if ( v75 != 65 )
          {
            v66 = 41;
LABEL_49:
            v74 = sub_16431D0(v78);
LABEL_30:
            v32 = sub_16982C0();
            v33 = a4[4];
            v72 = (__int64)v32;
            if ( (void *)v33 == v32 )
              sub_169C4E0(&v85, (__int64)v32);
            else
              sub_1698360((__int64)&v85, v33);
            v82 = v74;
            v65 = v74 - 1;
            v64 = 1LL << ((unsigned __int8)v74 - 1);
            v34 = ~v64;
            if ( v74 > 0x40 )
            {
              sub_16A4EF0((__int64)&v81, -1, 1);
              if ( v82 > 0x40 )
              {
                *(_QWORD *)(v81 + 8LL * (v65 >> 6)) &= v34;
LABEL_35:
                if ( v85 == (void *)v72 )
                  sub_169E6C0(&v85, (__int64)&v81, 1u, 0);
                else
                  sub_169A290((__int64)&v85, (__int64)&v81, 1, 0);
                if ( v82 > 0x40 && v81 )
                  j_j___libc_free_0_0(v81);
                if ( !(unsigned int)sub_14A9E40((__int64)v84, v76) )
                {
                  v35 = v66;
                  v36 = *(__int64 **)(a1[1] + 24);
                  if ( (unsigned int)(v66 - 40) <= 1 )
                    goto LABEL_74;
                  goto LABEL_42;
                }
                sub_127D120(&v85);
                v46 = a4[4];
                if ( v46 == v72 )
                  sub_169C4E0(&v85, v72);
                else
                  sub_1698360((__int64)&v85, v46);
                v82 = v74;
                if ( v74 > 0x40 )
                {
                  sub_16A4EF0((__int64)&v81, 0, 0);
                  if ( v82 > 0x40 )
                  {
                    *(_QWORD *)(v81 + 8LL * (v65 >> 6)) |= v64;
                    goto LABEL_86;
                  }
                }
                else
                {
                  v81 = 0;
                }
                v81 |= v64;
LABEL_86:
                if ( (void *)v72 == v85 )
                  sub_169E6C0(&v85, (__int64)&v81, 1u, 0);
                else
                  sub_169A290((__int64)&v85, (__int64)&v81, 1, 0);
                sub_135E100((__int64 *)&v81);
                if ( (unsigned int)sub_14A9E40((__int64)v84, v76) == 2 )
                {
                  v35 = v66;
                  v36 = *(__int64 **)(a1[1] + 24);
                  if ( (unsigned int)(v66 - 38) <= 1 )
                    goto LABEL_74;
LABEL_42:
                  if ( v35 != 33 )
                  {
LABEL_43:
                    v37 = sub_159C540(v36);
LABEL_44:
                    v28 = sub_170E100(a1, a2, v37, a5, a6, a7, a8, v38, v39, a11, a12);
LABEL_45:
                    sub_127D120(&v85);
                    return v28;
                  }
LABEL_74:
                  v37 = sub_159C4F0(v36);
                  goto LABEL_44;
                }
                sub_127D120(&v85);
                v51 = sub_15A40D0((unsigned __int64)a4, (__int64 **)v78, 0);
                goto LABEL_106;
              }
            }
            else
            {
              v81 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v74;
            }
            v81 &= v34;
            goto LABEL_35;
          }
          v66 = 37;
          break;
        case 7:
LABEL_60:
          v40 = a1;
          v41 = sub_159C4F0(*(__int64 **)(a1[1] + 24));
          return sub_170E100(v40, a2, v41, a5, a6, a7, a8, v42, v43, a11, a12);
        case 8:
LABEL_57:
          v40 = a1;
          v41 = sub_159C540(*(__int64 **)(a1[1] + 24));
          return sub_170E100(v40, a2, v41, a5, a6, a7, a8, v42, v43, a11, a12);
      }
      v74 = sub_16431D0(v78);
LABEL_63:
      v44 = sub_16982C0();
      v45 = a4[4];
      v72 = (__int64)v44;
      if ( (void *)v45 == v44 )
        sub_169C4E0(&v85, (__int64)v44);
      else
        sub_1698360((__int64)&v85, v45);
      v82 = v74;
      if ( v74 > 0x40 )
        sub_16A4EF0((__int64)&v81, -1, 1);
      else
        v81 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v74;
      if ( v85 == (void *)v72 )
        sub_169E6C0(&v85, (__int64)&v81, 0, 0);
      else
        sub_169A290((__int64)&v85, (__int64)&v81, 0, 0);
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
      if ( !(unsigned int)sub_14A9E40((__int64)v84, v76) )
      {
        v35 = v66;
        v36 = *(__int64 **)(a1[1] + 24);
        if ( (unsigned int)(v66 - 36) <= 1 )
          goto LABEL_74;
        goto LABEL_42;
      }
      sub_127D120(&v85);
      v47 = a4[4];
      if ( v47 == v72 )
        sub_169C4E0(&v85, v72);
      else
        sub_1698360((__int64)&v85, v47);
      v82 = v74;
      if ( v74 > 0x40 )
        sub_16A4EF0((__int64)&v81, 0, 0);
      else
        v81 = 0;
      if ( (void *)v72 == v85 )
        sub_169E6C0(&v85, (__int64)&v81, 1u, 0);
      else
        sub_169A290((__int64)&v85, (__int64)&v81, 1, 0);
      sub_135E100((__int64 *)&v81);
      if ( (unsigned int)sub_14A9E40((__int64)v84, v76) == 2 )
      {
        v36 = *(__int64 **)(a1[1] + 24);
        if ( (unsigned int)(v66 - 33) <= 2 )
        {
          v48 = sub_159C4F0(v36);
          v28 = sub_170E100(a1, a2, v48, a5, a6, a7, a8, v49, v50, a11, a12);
          goto LABEL_45;
        }
        goto LABEL_43;
      }
      sub_127D120(&v85);
      v51 = sub_15A4020((unsigned __int64)a4, (__int64 **)v78, 0);
LABEL_106:
      if ( v72 == a4[4] )
        v52 = a4[5] + 8LL;
      else
        v52 = (__int64)(a4 + 4);
      if ( (*(_BYTE *)(v52 + 18) & 7) != 3 )
      {
        v53 = (__int64 **)*a4;
        if ( v75 == 65 )
        {
          if ( a4 != (_QWORD *)sub_15A3EC0(v51, v53, 0) )
            goto LABEL_111;
        }
        else if ( a4 != (_QWORD *)sub_15A3F70(v51, v53, 0) )
        {
LABEL_111:
          switch ( v66 )
          {
            case '!':
              goto LABEL_60;
            case '"':
              if ( sub_173D840(v76) )
                goto LABEL_60;
              break;
            case '#':
              LOWORD(v66) = 34;
              if ( sub_173D840(v76) )
                goto LABEL_60;
              break;
            case '$':
              LOWORD(v66) = 37;
              if ( sub_173D840(v76) )
                goto LABEL_57;
              break;
            case '%':
              if ( sub_173D840(v76) )
                goto LABEL_57;
              break;
            case '&':
              v57 = !sub_173D840(v76);
              v62 = 39;
              if ( v57 )
                v62 = v66;
              LOWORD(v66) = v62;
              break;
            case '\'':
              v57 = !sub_173D840(v76);
              v61 = 38;
              if ( !v57 )
                v61 = v66;
              LOWORD(v66) = v61;
              break;
            case '(':
              v57 = !sub_173D840(v76);
              v58 = 41;
              if ( !v57 )
                v58 = v66;
              LOWORD(v66) = v58;
              break;
            case ')':
              v57 = !sub_173D840(v76);
              v63 = 40;
              if ( v57 )
                v63 = v66;
              LOWORD(v66) = v63;
              break;
            default:
              goto LABEL_57;
          }
        }
      }
      v59 = *(_QWORD *)sub_13CF970(a3);
      LOWORD(v86) = 257;
      v60 = sub_1648A60(56, 2u);
      v28 = (__int64)v60;
      if ( v60 )
        sub_17582E0((__int64)v60, v66, v59, v51, (__int64)v84);
      return v28;
    }
    v30 = (__int64)(a4 + 4);
    v71 = sub_16982C0();
    if ( (void *)a4[4] == v71 )
      v30 = a4[5] + 8LL;
    v31 = sub_169C2F0(v30);
    if ( v31 == 0x7FFFFFFF )
    {
      v54 = a4[4];
      if ( (void *)v54 == v71 )
        sub_169C580(&v85, (__int64)v71);
      else
        sub_1698390((__int64)&v85, v54);
      if ( v71 == v85 )
        sub_16A21B0((__int64)&v85, 0, *(double *)a5.m128_u64, a6, a7);
      else
        sub_169B370((__int64)&v85, 0);
      v55 = &v85;
      if ( v71 == v85 )
        v55 = (void **)(v86 + 8);
      v56 = sub_169C2F0((__int64)v55);
      sub_127D120(&v85);
      if ( v18 - (v75 != 65) <= v56 )
        goto LABEL_7;
    }
    else if ( v15 > v31 || v18 - (v75 != 65) < v31 )
    {
      goto LABEL_7;
    }
  }
  return 0;
}
