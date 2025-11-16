// Function: sub_14CF800
// Address: 0x14cf800
//
__int64 __fastcall sub_14CF800(__int64 a1, _BYTE *a2, _DWORD *a3, _QWORD *a4, unsigned __int64 *a5, unsigned int a6)
{
  _BYTE *v6; // r15
  unsigned __int8 v11; // al
  __int64 result; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  int v15; // eax
  bool v16; // al
  unsigned int v17; // r15d
  __int64 v18; // rax
  unsigned int v19; // eax
  bool v20; // al
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r15
  bool v24; // cc
  unsigned int v25; // ecx
  unsigned __int64 v26; // rdx
  unsigned int v27; // r15d
  unsigned int v28; // ecx
  unsigned __int64 v29; // rdx
  unsigned int v30; // r15d
  unsigned int v31; // eax
  bool v32; // al
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // r15
  unsigned int v36; // edx
  unsigned int v37; // ecx
  unsigned int v38; // r15d
  __int64 v39; // rax
  unsigned int v40; // ecx
  unsigned int v41; // r15d
  unsigned int v42; // r15d
  int v43; // eax
  int v44; // eax
  _QWORD *v45; // rax
  _QWORD *v46; // rax
  unsigned int v47; // eax
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rdi
  unsigned int v50; // [rsp+0h] [rbp-60h]
  __int64 v51; // [rsp+0h] [rbp-60h]
  unsigned int v52; // [rsp+0h] [rbp-60h]
  unsigned __int64 v53; // [rsp+0h] [rbp-60h]
  unsigned __int64 v54; // [rsp+0h] [rbp-60h]
  unsigned int v55; // [rsp+0h] [rbp-60h]
  unsigned int v56; // [rsp+0h] [rbp-60h]
  unsigned int v57; // [rsp+0h] [rbp-60h]
  __int64 v58; // [rsp+0h] [rbp-60h]
  unsigned int v59; // [rsp+0h] [rbp-60h]
  __int64 v60; // [rsp+0h] [rbp-60h]
  __int64 v61; // [rsp+0h] [rbp-60h]
  unsigned __int64 v62; // [rsp+0h] [rbp-60h]
  bool v63; // [rsp+0h] [rbp-60h]
  unsigned __int64 v64; // [rsp+0h] [rbp-60h]
  bool v65; // [rsp+0h] [rbp-60h]
  unsigned __int64 v67; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v68; // [rsp+18h] [rbp-48h]
  unsigned __int64 v69; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+28h] [rbp-38h]

  v6 = a2 + 24;
  v11 = a2[16];
  if ( v11 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      return 0;
    if ( v11 > 0x10u )
      return 0;
    v13 = sub_15A1020(a2);
    if ( !v13 )
      return 0;
    v6 = (_BYTE *)(v13 + 24);
    if ( *(_BYTE *)(v13 + 16) != 13 )
      return 0;
  }
  switch ( *a3 )
  {
    case '"':
      v68 = *((_DWORD *)v6 + 2);
      if ( v68 > 0x40 )
        sub_16A4FD0(&v67, v6);
      else
        v67 = *(_QWORD *)v6;
      sub_16A7490(&v67, 1);
      v19 = v68;
      v68 = 0;
      v70 = v19;
      v69 = v67;
      if ( v19 > 0x40 )
      {
        v64 = v67;
        v20 = (unsigned int)sub_16A5940(&v69) == 1;
        if ( v64 )
        {
          v49 = v64;
          v65 = v20;
          j_j___libc_free_0_0(v49);
          v20 = v65;
          if ( v68 > 0x40 )
          {
            if ( v67 )
            {
              j_j___libc_free_0_0(v67);
              v20 = v65;
            }
          }
        }
LABEL_25:
        if ( !v20 )
          return 0;
        v21 = *((_DWORD *)v6 + 2);
        v70 = v21;
        if ( v21 > 0x40 )
        {
          sub_16A4FD0(&v69, v6);
          v21 = v70;
          if ( v70 > 0x40 )
          {
            sub_16A8F40(&v69);
            v21 = v70;
            v23 = v69;
LABEL_29:
            v24 = *((_DWORD *)a5 + 2) <= 0x40u;
            v70 = 0;
            if ( v24 || !*a5 )
            {
              *a5 = v23;
              *((_DWORD *)a5 + 2) = v21;
              goto LABEL_34;
            }
            v52 = v21;
            j_j___libc_free_0_0(*a5);
            v24 = v70 <= 0x40;
            *a5 = v23;
            *((_DWORD *)a5 + 2) = v52;
            if ( v24 )
              goto LABEL_34;
            goto LABEL_32;
          }
          v22 = v69;
        }
        else
        {
          v22 = *(_QWORD *)v6;
        }
        v23 = ~v22 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21);
        v69 = v23;
        goto LABEL_29;
      }
      result = 0;
      if ( v67 )
      {
        v20 = (v67 & (v67 - 1)) == 0;
        goto LABEL_25;
      }
      return result;
    case '#':
      if ( *((_DWORD *)v6 + 2) <= 0x40u )
      {
        if ( !*(_QWORD *)v6 )
          return 0;
        if ( (*(_QWORD *)v6 & (*(_QWORD *)v6 - 1LL)) != 0 )
          return 0;
      }
      else if ( (unsigned int)sub_16A5940(v6) != 1 )
      {
        return 0;
      }
      v25 = *((_DWORD *)v6 + 2);
      v70 = v25;
      if ( v25 <= 0x40 )
      {
        v26 = *(_QWORD *)v6;
LABEL_39:
        v69 = ~v26 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v25);
        goto LABEL_40;
      }
      sub_16A4FD0(&v69, v6);
      LOBYTE(v25) = v70;
      if ( v70 <= 0x40 )
      {
        v26 = v69;
        goto LABEL_39;
      }
      sub_16A8F40(&v69);
LABEL_40:
      sub_16A7400(&v69);
      v24 = *((_DWORD *)a5 + 2) <= 0x40u;
      v27 = v70;
      v70 = 0;
      if ( v24 || !*a5 )
      {
        *a5 = v69;
        *((_DWORD *)a5 + 2) = v27;
      }
      else
      {
        v53 = v69;
        j_j___libc_free_0_0(*a5);
        v24 = v70 <= 0x40;
        *((_DWORD *)a5 + 2) = v27;
        *a5 = v53;
        if ( !v24 )
        {
LABEL_32:
          if ( v69 )
            j_j___libc_free_0_0(v69);
        }
      }
LABEL_34:
      *a3 = 33;
      goto LABEL_68;
    case '$':
      if ( *((_DWORD *)v6 + 2) <= 0x40u )
      {
        if ( !*(_QWORD *)v6 )
          return 0;
        if ( (*(_QWORD *)v6 & (*(_QWORD *)v6 - 1LL)) != 0 )
          return 0;
      }
      else if ( (unsigned int)sub_16A5940(v6) != 1 )
      {
        return 0;
      }
      v28 = *((_DWORD *)v6 + 2);
      v70 = v28;
      if ( v28 <= 0x40 )
      {
        v29 = *(_QWORD *)v6;
LABEL_48:
        v69 = ~v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28);
        goto LABEL_49;
      }
      sub_16A4FD0(&v69, v6);
      LOBYTE(v28) = v70;
      if ( v70 <= 0x40 )
      {
        v29 = v69;
        goto LABEL_48;
      }
      sub_16A8F40(&v69);
LABEL_49:
      sub_16A7400(&v69);
      v24 = *((_DWORD *)a5 + 2) <= 0x40u;
      v30 = v70;
      v70 = 0;
      if ( v24 || !*a5 )
      {
        *a5 = v69;
        *((_DWORD *)a5 + 2) = v30;
      }
      else
      {
        v54 = v69;
        j_j___libc_free_0_0(*a5);
        v24 = v70 <= 0x40;
        *((_DWORD *)a5 + 2) = v30;
        *a5 = v54;
        if ( !v24 )
          goto LABEL_65;
      }
      goto LABEL_67;
    case '%':
      v68 = *((_DWORD *)v6 + 2);
      if ( v68 > 0x40 )
        sub_16A4FD0(&v67, v6);
      else
        v67 = *(_QWORD *)v6;
      sub_16A7490(&v67, 1);
      v31 = v68;
      v68 = 0;
      v70 = v31;
      v69 = v67;
      if ( v31 > 0x40 )
      {
        v62 = v67;
        v32 = (unsigned int)sub_16A5940(&v69) == 1;
        if ( v62 )
        {
          v48 = v62;
          v63 = v32;
          j_j___libc_free_0_0(v48);
          v32 = v63;
          if ( v68 > 0x40 )
          {
            if ( v67 )
            {
              j_j___libc_free_0_0(v67);
              v32 = v63;
            }
          }
        }
      }
      else
      {
        result = 0;
        if ( !v67 )
          return result;
        v32 = (v67 & (v67 - 1)) == 0;
      }
      if ( !v32 )
        return 0;
      v33 = *((_DWORD *)v6 + 2);
      v70 = v33;
      if ( v33 <= 0x40 )
      {
        v34 = *(_QWORD *)v6;
LABEL_61:
        v35 = ~v34 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v33);
        v69 = v35;
        goto LABEL_62;
      }
      sub_16A4FD0(&v69, v6);
      v33 = v70;
      if ( v70 <= 0x40 )
      {
        v34 = v69;
        goto LABEL_61;
      }
      sub_16A8F40(&v69);
      v33 = v70;
      v35 = v69;
LABEL_62:
      v24 = *((_DWORD *)a5 + 2) <= 0x40u;
      v70 = 0;
      if ( v24 || !*a5 )
      {
        *a5 = v35;
        *((_DWORD *)a5 + 2) = v33;
      }
      else
      {
        v55 = v33;
        j_j___libc_free_0_0(*a5);
        v24 = v70 <= 0x40;
        *a5 = v35;
        *((_DWORD *)a5 + 2) = v55;
        if ( !v24 )
        {
LABEL_65:
          if ( v69 )
            j_j___libc_free_0_0(v69);
        }
      }
LABEL_67:
      *a3 = 32;
      goto LABEL_68;
    case '&':
      v36 = *((_DWORD *)v6 + 2);
      if ( v36 <= 0x40 )
      {
        if ( *(_QWORD *)v6 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) )
          return 0;
        v70 = *((_DWORD *)v6 + 2);
        v69 = 0;
        v39 = 1LL << ((unsigned __int8)v36 - 1);
      }
      else
      {
        v56 = *((_DWORD *)v6 + 2);
        if ( v56 != (unsigned int)sub_16A58F0(v6) )
          return 0;
        v42 = v56 - 1;
        v70 = v56;
        v61 = 1LL << ((unsigned __int8)v56 - 1);
        sub_16A4EF0(&v69, 0, 0);
        v39 = v61;
        if ( v70 > 0x40 )
        {
          *(_QWORD *)(v69 + 8LL * (v42 >> 6)) |= v61;
          goto LABEL_90;
        }
      }
      goto LABEL_89;
    case '\'':
      v37 = *((_DWORD *)v6 + 2);
      if ( v37 <= 0x40 )
      {
        if ( *(_QWORD *)v6 )
          return 0;
        v69 = 0;
        v70 = v37;
        v39 = 1LL << ((unsigned __int8)v37 - 1);
      }
      else
      {
        v57 = *((_DWORD *)v6 + 2);
        if ( v57 != (unsigned int)sub_16A57B0(v6) )
          return 0;
        v38 = v57 - 1;
        v70 = v57;
        v58 = 1LL << ((unsigned __int8)v57 - 1);
        sub_16A4EF0(&v69, 0, 0);
        v39 = v58;
        if ( v70 > 0x40 )
        {
          *(_QWORD *)(v69 + 8LL * (v38 >> 6)) |= v58;
          goto LABEL_90;
        }
      }
LABEL_89:
      v69 |= v39;
LABEL_90:
      if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
        j_j___libc_free_0_0(*a5);
      *a5 = v69;
      *((_DWORD *)a5 + 2) = v70;
      *a3 = 32;
      goto LABEL_68;
    case '(':
      v40 = *((_DWORD *)v6 + 2);
      if ( v40 <= 0x40 )
      {
        if ( !*(_QWORD *)v6 )
        {
          v70 = *((_DWORD *)v6 + 2);
          v18 = 1LL << ((unsigned __int8)v40 - 1);
LABEL_98:
          v69 = 0;
          goto LABEL_99;
        }
        return 0;
      }
      v59 = *((_DWORD *)v6 + 2);
      if ( v59 != (unsigned int)sub_16A57B0(v6) )
        return 0;
      v41 = v59 - 1;
      v70 = v59;
      v60 = 1LL << ((unsigned __int8)v59 - 1);
      sub_16A4EF0(&v69, 0, 0);
      v18 = v60;
      if ( v70 > 0x40 )
      {
        *(_QWORD *)(v69 + 8LL * (v41 >> 6)) |= v60;
        goto LABEL_16;
      }
LABEL_99:
      v69 |= v18;
LABEL_16:
      if ( *((_DWORD *)a5 + 2) > 0x40u && *a5 )
        j_j___libc_free_0_0(*a5);
      *a5 = v69;
      *((_DWORD *)a5 + 2) = v70;
      *a3 = 33;
LABEL_68:
      if ( !(_BYTE)a6 )
      {
LABEL_69:
        *a4 = a1;
        return 1;
      }
      v43 = *(unsigned __int8 *)(a1 + 16);
      if ( (unsigned __int8)v43 > 0x17u )
      {
        v44 = v43 - 24;
      }
      else
      {
        if ( (_BYTE)v43 != 5 )
          goto LABEL_69;
        v44 = *(unsigned __int16 *)(a1 + 18);
      }
      if ( v44 != 36 )
        goto LABEL_69;
      v45 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
          ? *(_QWORD **)(a1 - 8)
          : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v46 = (_QWORD *)*v45;
      if ( !v46 )
        goto LABEL_69;
      *a4 = v46;
      v47 = sub_16431D0(*v46);
      sub_16A5C50(&v69, a5, v47);
      if ( *((_DWORD *)a5 + 2) > 0x40u )
      {
        if ( *a5 )
          j_j___libc_free_0_0(*a5);
      }
      *a5 = v69;
      *((_DWORD *)a5 + 2) = v70;
      return a6;
    case ')':
      v14 = *((_DWORD *)v6 + 2);
      if ( v14 <= 0x40 )
      {
        v16 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) == *(_QWORD *)v6;
      }
      else
      {
        v50 = *((_DWORD *)v6 + 2);
        v15 = sub_16A58F0(v6);
        v14 = v50;
        v16 = v50 == v15;
      }
      if ( !v16 )
        return 0;
      v17 = v14 - 1;
      v70 = v14;
      v18 = 1LL << ((unsigned __int8)v14 - 1);
      if ( v14 <= 0x40 )
        goto LABEL_98;
      v51 = 1LL << ((unsigned __int8)v14 - 1);
      sub_16A4EF0(&v69, 0, 0);
      v18 = v51;
      if ( v70 <= 0x40 )
        goto LABEL_99;
      *(_QWORD *)(v69 + 8LL * (v17 >> 6)) |= v51;
      goto LABEL_16;
    default:
      return 0;
  }
}
