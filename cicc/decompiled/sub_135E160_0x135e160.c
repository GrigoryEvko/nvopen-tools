// Function: sub_135E160
// Address: 0x135e160
//
__int64 __fastcall sub_135E160(
        __int64 a1,
        _DWORD *a2,
        unsigned int *a3,
        int *a4,
        int *a5,
        int a6,
        int a7,
        __int64 a8,
        __int64 a9,
        _BYTE *a10,
        _BYTE *a11)
{
  char v13; // al
  __int64 v14; // r8
  __int64 v18; // rsi
  int v19; // esi
  int v20; // r11d
  unsigned int v21; // ecx
  int v22; // r9d
  unsigned int v23; // ecx
  unsigned int v24; // r15d
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  unsigned int v28; // r12d
  unsigned __int64 v29; // rsi
  unsigned int v30; // ecx
  char v31; // al
  unsigned int v32; // eax
  unsigned int v33; // ecx
  unsigned int v34; // r15d
  bool v35; // cc
  __int64 v36; // rdx
  char v37; // al
  int v38; // eax
  unsigned int v39; // ecx
  __int64 v40; // rdx
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // rdi
  int v44; // eax
  int v45; // [rsp+8h] [rbp-88h]
  int v46; // [rsp+Ch] [rbp-84h]
  int v48; // [rsp+18h] [rbp-78h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  unsigned int v50; // [rsp+20h] [rbp-70h]
  __int64 v51; // [rsp+20h] [rbp-70h]
  __int64 v52; // [rsp+20h] [rbp-70h]
  int v53; // [rsp+20h] [rbp-70h]
  int v54; // [rsp+20h] [rbp-70h]
  int v55; // [rsp+20h] [rbp-70h]
  _QWORD *v57; // [rsp+28h] [rbp-68h]
  __int64 v58; // [rsp+28h] [rbp-68h]
  __int64 v59; // [rsp+28h] [rbp-68h]
  __int64 v60; // [rsp+28h] [rbp-68h]
  __int64 v61; // [rsp+28h] [rbp-68h]
  int v62; // [rsp+28h] [rbp-68h]
  int v63; // [rsp+28h] [rbp-68h]
  int v64; // [rsp+28h] [rbp-68h]
  __int64 v65; // [rsp+28h] [rbp-68h]
  __int64 v66; // [rsp+28h] [rbp-68h]
  __int64 v67; // [rsp+28h] [rbp-68h]
  __int64 v68; // [rsp+28h] [rbp-68h]
  __int64 v69; // [rsp+28h] [rbp-68h]
  __int64 v70; // [rsp+28h] [rbp-68h]
  __int64 v71; // [rsp+28h] [rbp-68h]
  __int64 v72; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v73; // [rsp+38h] [rbp-58h]
  __int64 v74; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v75; // [rsp+48h] [rbp-48h]
  unsigned __int64 *v76; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v77; // [rsp+58h] [rbp-38h]

  if ( a7 == 6 )
    goto LABEL_12;
  v13 = *(_BYTE *)(a1 + 16);
  if ( v13 == 13 )
  {
    sub_16A5DD0(&v76, a1 + 24, a3[2]);
    sub_16A7200(a3, &v76);
    if ( v77 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
    return a1;
  }
  if ( (unsigned __int8)(v13 - 35) > 0x11u )
  {
    if ( (unsigned __int8)(v13 - 61) <= 1u )
    {
      v57 = *(_QWORD **)(a1 - 24);
      v50 = sub_1643030(*(_QWORD *)a1);
      v48 = sub_1643030(*v57);
      v45 = *a4;
      v46 = *a5;
      v14 = sub_135E160(
              (_DWORD)v57,
              (_DWORD)a2,
              (_DWORD)a3,
              (_DWORD)a4,
              (_DWORD)a5,
              a6,
              a7 + 1,
              a8,
              a9,
              (__int64)a10,
              (__int64)a11);
      v22 = v50 - v48;
      if ( *(_BYTE *)(a1 + 16) != 62 || *a4 )
      {
        if ( !*a11 )
        {
          v33 = a2[2];
          if ( v33 > 0x40 )
          {
            **(_QWORD **)a2 = 1;
            memset(
              (void *)(*(_QWORD *)a2 + 8LL),
              0,
              8 * (unsigned int)(((unsigned __int64)(unsigned int)a2[2] + 63) >> 6) - 8);
            v22 = v50 - v48;
          }
          else
          {
            *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v33) & 1;
          }
          if ( a3[2] > 0x40 )
          {
            v53 = v22;
            **(_QWORD **)a3 = 0;
            memset((void *)(*(_QWORD *)a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)a3[2] + 63) >> 6) - 8);
            v22 = v53;
          }
          else
          {
            *(_QWORD *)a3 = 0;
          }
          v14 = (__int64)v57;
          *a4 = v45;
          *a5 = v46;
        }
        *a4 += v22;
      }
      else
      {
        v49 = v14;
        if ( *a10 )
        {
          v62 = v22;
          v34 = a3[2];
          sub_16A5A50(&v72, a3);
          sub_16A5B10(&v74, &v72, v50);
          sub_16A5DD0(&v76, &v74, v34);
          v22 = v62;
          v14 = v49;
          if ( a3[2] > 0x40 && *(_QWORD *)a3 )
          {
            j_j___libc_free_0_0(*(_QWORD *)a3);
            v14 = v49;
            v22 = v62;
          }
          v35 = v75 <= 0x40;
          *(_QWORD *)a3 = v76;
          a3[2] = v77;
          if ( !v35 && v74 )
          {
            v51 = v14;
            v63 = v22;
            j_j___libc_free_0_0(v74);
            v14 = v51;
            v22 = v63;
          }
          if ( v73 > 0x40 && v72 )
          {
            v52 = v14;
            v64 = v22;
            j_j___libc_free_0_0(v72);
            v14 = v52;
            v22 = v64;
          }
          v46 = *a5;
        }
        else
        {
          v39 = a2[2];
          if ( v39 > 0x40 )
          {
            v55 = v22;
            **(_QWORD **)a2 = 1;
            memset(
              (void *)(*(_QWORD *)a2 + 8LL),
              0,
              8 * (unsigned int)(((unsigned __int64)(unsigned int)a2[2] + 63) >> 6) - 8);
            v22 = v55;
          }
          else
          {
            *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v39) & 1;
          }
          if ( a3[2] > 0x40 )
          {
            v54 = v22;
            **(_QWORD **)a3 = 0;
            memset((void *)(*(_QWORD *)a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)a3[2] + 63) >> 6) - 8);
            v22 = v54;
          }
          else
          {
            *(_QWORD *)a3 = 0;
          }
          v14 = (__int64)v57;
          *a4 = v45;
        }
        *a5 = v46 + v22;
      }
      return v14;
    }
LABEL_12:
    v21 = a2[2];
    if ( v21 > 0x40 )
    {
      **(_QWORD **)a2 = 1;
      memset(
        (void *)(*(_QWORD *)a2 + 8LL),
        0,
        8 * (unsigned int)(((unsigned __int64)(unsigned int)a2[2] + 63) >> 6) - 8);
      if ( a3[2] <= 0x40 )
        goto LABEL_14;
    }
    else
    {
      *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v21) & 1;
      if ( a3[2] <= 0x40 )
      {
LABEL_14:
        *(_QWORD *)a3 = 0;
        return a1;
      }
    }
    **(_QWORD **)a3 = 0;
    memset((void *)(*(_QWORD *)a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)a3[2] + 63) >> 6) - 8);
    return a1;
  }
  v18 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v18 + 16) != 13 )
    goto LABEL_12;
  sub_16A5DD0(&v76, v18 + 24, a3[2]);
  v19 = v18 + 24;
  v20 = a6;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
      goto LABEL_28;
    case '%':
      v60 = sub_135E160(
              *(_QWORD *)(a1 - 48),
              (_DWORD)a2,
              (_DWORD)a3,
              (_DWORD)a4,
              (_DWORD)a5,
              a6,
              a7 + 1,
              a8,
              a9,
              (__int64)a10,
              (__int64)a11);
      sub_16A7590(a3, &v76);
      v14 = v60;
      goto LABEL_29;
    case '\'':
      v61 = sub_135E160(
              *(_QWORD *)(a1 - 48),
              (_DWORD)a2,
              (_DWORD)a3,
              (_DWORD)a4,
              (_DWORD)a5,
              a6,
              a7 + 1,
              a8,
              a9,
              (__int64)a10,
              (__int64)a11);
      sub_16A7C10(a3, &v76);
      sub_16A7C10(a2, &v76);
      v14 = v61;
      goto LABEL_29;
    case '/':
      v26 = sub_135E160(
              *(_QWORD *)(a1 - 48),
              (_DWORD)a2,
              (_DWORD)a3,
              (_DWORD)a4,
              (_DWORD)a5,
              a6,
              a7 + 1,
              a8,
              a9,
              (__int64)a10,
              (__int64)a11);
      v27 = a3[2];
      v24 = v77;
      v14 = v26;
      v28 = a3[2];
      if ( v77 > 0x40 )
      {
        v66 = v26;
        v38 = sub_16A57B0(&v76);
        v30 = a2[2];
        v14 = v66;
        if ( v24 - v38 > 0x40 )
          goto LABEL_35;
        v29 = *v76;
      }
      else
      {
        v29 = (unsigned __int64)v76;
        v30 = a2[2];
      }
      if ( v27 < v29 || v30 < v29 )
      {
LABEL_35:
        if ( v30 > 0x40 )
        {
          v68 = v14;
          **(_QWORD **)a2 = 1;
          memset(
            (void *)(*(_QWORD *)a2 + 8LL),
            0,
            8 * (unsigned int)(((unsigned __int64)(unsigned int)a2[2] + 63) >> 6) - 8);
          v14 = v68;
        }
        else
        {
          *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v30) & 1;
        }
        if ( a3[2] > 0x40 )
        {
          v67 = v14;
          **(_QWORD **)a3 = 0;
          memset((void *)(*(_QWORD *)a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)a3[2] + 63) >> 6) - 8);
          v24 = v77;
          v14 = v67;
        }
        else
        {
          *(_QWORD *)a3 = 0;
          v24 = v77;
        }
        goto LABEL_25;
      }
      if ( v28 > 0x40 )
      {
        v71 = v14;
        sub_16A7DC0(a3, v29);
        v24 = v77;
        v14 = v71;
      }
      else
      {
        v40 = 0;
        if ( (_DWORD)v29 != v28 )
          v40 = *(_QWORD *)a3 << v29;
        *(_QWORD *)a3 = v40 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28);
      }
      if ( v24 > 0x40 )
      {
        v69 = v14;
        v44 = sub_16A57B0(&v76);
        v14 = v69;
        v41 = -1;
        if ( v24 - v44 <= 0x40 )
          v41 = *v76;
      }
      else
      {
        v41 = (__int64)v76;
      }
      v42 = a2[2];
      if ( v42 > 0x40 )
      {
        v70 = v14;
        sub_16A7DC0(a2, v41);
        v24 = v77;
        v14 = v70;
      }
      else
      {
        v43 = 0;
        if ( (_DWORD)v41 != v42 )
          v43 = *(_QWORD *)a2 << v41;
        *(_QWORD *)a2 = v43 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v42);
      }
      *a11 = 0;
      *a10 = 0;
LABEL_25:
      if ( v24 > 0x40 && v76 )
      {
        v58 = v14;
        j_j___libc_free_0_0(v76);
        v14 = v58;
      }
      break;
    case '3':
      v31 = sub_14C1670(*(_QWORD *)(a1 - 48), v19, a6, 0, a8, a1, a9);
      v20 = a6;
      if ( v31 )
      {
LABEL_28:
        v59 = sub_135E160(
                *(_QWORD *)(a1 - 48),
                (_DWORD)a2,
                (_DWORD)a3,
                (_DWORD)a4,
                (_DWORD)a5,
                v20,
                a7 + 1,
                a8,
                a9,
                (__int64)a10,
                (__int64)a11);
        sub_16A7200(a3, &v76);
        v14 = v59;
LABEL_29:
        v25 = *(unsigned __int8 *)(a1 + 16);
        if ( (unsigned __int8)v25 <= 0x2Fu && (v36 = 0x80A800000000LL, _bittest64(&v36, v25)) )
        {
          v65 = v14;
          *a11 &= sub_15F2370(a1);
          v37 = sub_15F2380(a1);
          v24 = v77;
          v14 = v65;
          *a10 &= v37;
        }
        else
        {
          v24 = v77;
        }
      }
      else
      {
        v32 = a2[2];
        if ( v32 > 0x40 )
        {
LABEL_67:
          **(_QWORD **)a2 = 1;
          memset(
            (void *)(*(_QWORD *)a2 + 8LL),
            0,
            8 * (unsigned int)(((unsigned __int64)(unsigned int)a2[2] + 63) >> 6) - 8);
        }
        else
        {
          *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & 1;
        }
LABEL_22:
        if ( a3[2] > 0x40 )
        {
          **(_QWORD **)a3 = 0;
          memset((void *)(*(_QWORD *)a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)a3[2] + 63) >> 6) - 8);
        }
        else
        {
          *(_QWORD *)a3 = 0;
        }
        v24 = v77;
        v14 = a1;
      }
      goto LABEL_25;
    default:
      v23 = a2[2];
      if ( v23 > 0x40 )
        goto LABEL_67;
      *(_QWORD *)a2 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v23) & 1;
      goto LABEL_22;
  }
  return v14;
}
