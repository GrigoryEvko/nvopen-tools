// Function: sub_13AB330
// Address: 0x13ab330
//
_BOOL8 __fastcall sub_13AB330(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // r12
  __int64 v8; // rax
  __int16 v9; // dx
  __int64 v10; // rsi
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // rax
  __int16 v16; // dx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // r14
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 v26; // r13
  __int64 v27; // rbx
  __int64 v28; // rax
  __int16 v29; // dx
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v33; // rbx
  __int64 v34; // rbx
  __int64 v35; // rsi
  __int64 v36; // rax
  __int16 v37; // dx
  __int64 v38; // rsi
  unsigned int v39; // edx
  int v40; // eax
  __int64 v41; // rbx
  __int64 v42; // rax
  __int16 v43; // dx
  __int64 v44; // rax
  __int16 v45; // dx
  _BYTE *v46; // rax
  __int64 v47; // [rsp+8h] [rbp-128h]
  __int64 v48; // [rsp+10h] [rbp-120h]
  __int64 v49; // [rsp+20h] [rbp-110h]
  unsigned int v50; // [rsp+40h] [rbp-F0h]
  _QWORD **v52; // [rsp+50h] [rbp-E0h]
  bool v53; // [rsp+60h] [rbp-D0h]
  __int64 v54; // [rsp+60h] [rbp-D0h]
  _QWORD *v57; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v58; // [rsp+88h] [rbp-A8h]
  _QWORD *v59; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v60; // [rsp+98h] [rbp-98h]
  __int64 v61[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v62[2]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v63; // [rsp+C0h] [rbp-70h] BYREF
  unsigned int v64; // [rsp+C8h] [rbp-68h]
  _QWORD *v65; // [rsp+D0h] [rbp-60h] BYREF
  unsigned int v66; // [rsp+D8h] [rbp-58h]
  _QWORD *v67; // [rsp+E0h] [rbp-50h] BYREF
  unsigned int v68; // [rsp+E8h] [rbp-48h]
  _QWORD *v69; // [rsp+F0h] [rbp-40h] BYREF
  unsigned int v70; // [rsp+F8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_1456040(a2);
  v58 = sub_1456C90(v4, v5);
  v6 = v58;
  if ( v58 > 0x40 )
  {
    sub_16A4EF0(&v57, 0, 0);
    v7 = a2;
    if ( *(_WORD *)(a2 + 24) == 7 )
      goto LABEL_3;
    goto LABEL_59;
  }
  v7 = a2;
  v57 = 0;
  if ( *(_WORD *)(a2 + 24) != 7 )
  {
LABEL_59:
    v14 = a3;
    v7 = a2;
    if ( *(_WORD *)(a3 + 24) == 7 )
      goto LABEL_30;
    goto LABEL_60;
  }
  do
  {
LABEL_3:
    v8 = sub_13A5BC0((_QWORD *)v7, *(_QWORD *)(a1 + 8));
    v9 = *(_WORD *)(v8 + 24);
    if ( v9 )
    {
      if ( v9 != 5 || (v8 = **(_QWORD **)(v8 + 32), *(_WORD *)(v8 + 24)) )
      {
LABEL_72:
        v53 = 0;
        goto LABEL_96;
      }
    }
    v10 = *(_QWORD *)(v8 + 32);
    v11 = *(_DWORD *)(v10 + 32);
    v64 = v11;
    if ( v11 <= 0x40 )
    {
      v63 = *(_QWORD *)(v10 + 24);
      v12 = 1LL << ((unsigned __int8)v11 - 1);
LABEL_6:
      v13 = v63;
      if ( (v12 & v63) != 0 )
      {
        v70 = v11;
        goto LABEL_8;
      }
      v66 = v11;
      v65 = (_QWORD *)v63;
      goto LABEL_14;
    }
    sub_16A4FD0(&v63, v10 + 24);
    v11 = v64;
    v12 = 1LL << ((unsigned __int8)v64 - 1);
    if ( v64 <= 0x40 )
      goto LABEL_6;
    if ( (*(_QWORD *)(v63 + 8LL * ((v64 - 1) >> 6)) & v12) != 0 )
    {
      v70 = v64;
      sub_16A4FD0(&v69, &v63);
      LOBYTE(v11) = v70;
      if ( v70 > 0x40 )
      {
        sub_16A8F40(&v69);
LABEL_9:
        sub_16A7400(&v69);
        v66 = v70;
        v65 = v69;
        v68 = v58;
        if ( v58 <= 0x40 )
          goto LABEL_15;
        goto LABEL_10;
      }
      v13 = (__int64)v69;
LABEL_8:
      v69 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v13);
      goto LABEL_9;
    }
    v66 = v64;
    sub_16A4FD0(&v65, &v63);
LABEL_14:
    v68 = v58;
    if ( v58 <= 0x40 )
    {
LABEL_15:
      v67 = v57;
      goto LABEL_16;
    }
LABEL_10:
    sub_16A4FD0(&v67, &v57);
LABEL_16:
    sub_16A9A30(&v69, &v67, &v65);
    if ( v58 > 0x40 && v57 )
      j_j___libc_free_0_0(v57);
    v57 = v69;
    v58 = v70;
    if ( v68 > 0x40 && v67 )
      j_j___libc_free_0_0(v67);
    if ( v66 > 0x40 && v65 )
      j_j___libc_free_0_0(v65);
    v7 = **(_QWORD **)(v7 + 32);
    if ( v64 > 0x40 && v63 )
      j_j___libc_free_0_0(v63);
  }
  while ( *(_WORD *)(v7 + 24) == 7 );
  v14 = a3;
  if ( *(_WORD *)(a3 + 24) == 7 )
  {
    while ( 1 )
    {
LABEL_30:
      v15 = sub_13A5BC0((_QWORD *)v14, *(_QWORD *)(a1 + 8));
      v16 = *(_WORD *)(v15 + 24);
      if ( v16 )
      {
        if ( v16 != 5 )
          goto LABEL_72;
        v15 = **(_QWORD **)(v15 + 32);
        if ( *(_WORD *)(v15 + 24) )
          goto LABEL_72;
      }
      v17 = *(_QWORD *)(v15 + 32);
      v18 = *(_DWORD *)(v17 + 32);
      v64 = v18;
      if ( v18 <= 0x40 )
        break;
      sub_16A4FD0(&v63, v17 + 24);
      v18 = v64;
      v19 = 1LL << ((unsigned __int8)v64 - 1);
      if ( v64 <= 0x40 )
        goto LABEL_33;
      if ( (*(_QWORD *)(v63 + 8LL * ((v64 - 1) >> 6)) & v19) != 0 )
      {
        v70 = v64;
        sub_16A4FD0(&v69, &v63);
        LOBYTE(v18) = v70;
        if ( v70 > 0x40 )
        {
          sub_16A8F40(&v69);
LABEL_36:
          sub_16A7400(&v69);
          v66 = v70;
          v65 = v69;
LABEL_37:
          v68 = v58;
          if ( v58 <= 0x40 )
            goto LABEL_38;
          goto LABEL_68;
        }
        v20 = (__int64)v69;
LABEL_35:
        v69 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v20);
        goto LABEL_36;
      }
      v66 = v64;
      sub_16A4FD0(&v65, &v63);
      v68 = v58;
      if ( v58 <= 0x40 )
      {
LABEL_38:
        v67 = v57;
        goto LABEL_39;
      }
LABEL_68:
      sub_16A4FD0(&v67, &v57);
LABEL_39:
      sub_16A9A30(&v69, &v67, &v65);
      if ( v58 > 0x40 && v57 )
        j_j___libc_free_0_0(v57);
      v57 = v69;
      v58 = v70;
      if ( v68 > 0x40 && v67 )
        j_j___libc_free_0_0(v67);
      if ( v66 > 0x40 && v65 )
        j_j___libc_free_0_0(v65);
      v14 = **(_QWORD **)(v14 + 32);
      if ( v64 > 0x40 && v63 )
        j_j___libc_free_0_0(v63);
      if ( *(_WORD *)(v14 + 24) != 7 )
      {
        v60 = v6;
        if ( v6 <= 0x40 )
          goto LABEL_53;
LABEL_61:
        sub_16A4EF0(&v59, 0, 0);
        goto LABEL_54;
      }
    }
    v63 = *(_QWORD *)(v17 + 24);
    v19 = 1LL << ((unsigned __int8)v18 - 1);
LABEL_33:
    v20 = v63;
    if ( (v63 & v19) == 0 )
    {
      v66 = v18;
      v65 = (_QWORD *)v63;
      goto LABEL_37;
    }
    v70 = v18;
    goto LABEL_35;
  }
LABEL_60:
  v60 = v6;
  v14 = a3;
  if ( v6 > 0x40 )
    goto LABEL_61;
LABEL_53:
  v59 = 0;
LABEL_54:
  v21 = sub_14806B0(*(_QWORD *)(a1 + 8), v14, v7, 0, 0);
  v22 = *(_WORD *)(v21 + 24);
  v23 = v21;
  if ( v22 )
  {
    if ( v22 != 4 )
      goto LABEL_92;
    v24 = *(_QWORD *)(v21 + 40);
    if ( !(_DWORD)v24 )
      goto LABEL_92;
    v25 = 0;
    v26 = v21;
    v54 = 0;
    v27 = 8LL * (unsigned int)v24;
    do
    {
      v28 = *(_QWORD *)(*(_QWORD *)(v26 + 32) + v25);
      v29 = *(_WORD *)(v28 + 24);
      if ( v29 )
      {
        if ( v29 != 5 )
          goto LABEL_92;
        v30 = **(_QWORD **)(v28 + 32);
        if ( *(_WORD *)(v30 + 24) )
          goto LABEL_92;
        v31 = *(_QWORD *)(v30 + 32);
        v64 = *(_DWORD *)(v31 + 32);
        if ( v64 > 0x40 )
          sub_16A4FD0(&v63, v31 + 24);
        else
          v63 = *(_QWORD *)(v31 + 24);
        sub_13A3E40((__int64)&v65, (__int64)&v63);
        v68 = v60;
        if ( v60 > 0x40 )
          sub_16A4FD0(&v67, &v59);
        else
          v67 = v59;
        sub_16A9A30(&v69, &v67, &v65);
        sub_13A3610((__int64 *)&v59, (__int64 *)&v69);
        sub_135E100((__int64 *)&v69);
        sub_135E100((__int64 *)&v67);
        sub_135E100((__int64 *)&v65);
        sub_135E100(&v63);
      }
      else
      {
        v54 = *(_QWORD *)(*(_QWORD *)(v26 + 32) + v25);
      }
      v25 += 8;
    }
    while ( v27 != v25 );
    if ( !v54 )
    {
LABEL_92:
      v53 = 0;
      goto LABEL_93;
    }
    v23 = v54;
  }
  sub_13A38D0((__int64)v61, *(_QWORD *)(v23 + 32) + 24LL);
  v53 = sub_13A38F0((__int64)v61, 0);
  if ( v53 )
  {
    v53 = 0;
    goto LABEL_57;
  }
  sub_13A38D0((__int64)&v67, (__int64)&v59);
  sub_13A38D0((__int64)&v65, (__int64)&v57);
  sub_16A9A30(&v69, &v65, &v67);
  sub_13A3610((__int64 *)&v57, (__int64 *)&v69);
  sub_135E100((__int64 *)&v69);
  sub_135E100((__int64 *)&v65);
  sub_135E100((__int64 *)&v67);
  sub_16AB4D0(v62, v61, &v57);
  if ( !sub_13A38F0((__int64)v62, 0) )
  {
    v53 = 1;
    goto LABEL_150;
  }
  v33 = a2;
  if ( *(_WORD *)(a2 + 24) != 7 )
    goto LABEL_150;
  while ( 2 )
  {
    v48 = **(_QWORD **)(v33 + 32);
    v52 = *(_QWORD ***)(v33 + 48);
    sub_13A36B0((__int64)&v57, (__int64 *)&v59);
    v47 = sub_13A5BC0((_QWORD *)v33, *(_QWORD *)(a1 + 8));
    v34 = a2;
    v49 = sub_14806B0(*(_QWORD *)(a1 + 8), v47, v47, 0, 0);
    while ( 1 )
    {
      v39 = v58;
      if ( v58 <= 0x40 )
      {
        if ( v57 == (_QWORD *)1 )
          goto LABEL_128;
      }
      else
      {
        v50 = v58;
        v40 = sub_16A57B0(&v57);
        v39 = v50;
        if ( v50 - v40 <= 0x40 && *v57 == 1 )
        {
LABEL_128:
          v41 = a3;
          v35 = *(_QWORD *)(a1 + 8);
          goto LABEL_137;
        }
      }
      v35 = *(_QWORD *)(a1 + 8);
      if ( *(_WORD *)(v34 + 24) != 7 )
        break;
      v36 = sub_13A5BC0((_QWORD *)v34, v35);
      if ( *(_QWORD ***)(v34 + 48) != v52 )
      {
        v37 = *(_WORD *)(v36 + 24);
        if ( v37 )
        {
          if ( v37 != 5 )
            goto LABEL_150;
          v36 = **(_QWORD **)(v36 + 32);
          if ( *(_WORD *)(v36 + 24) )
            goto LABEL_150;
        }
        v38 = *(_QWORD *)(v36 + 32);
        v64 = *(_DWORD *)(v38 + 32);
        if ( v64 > 0x40 )
          sub_16A4FD0(&v63, v38 + 24);
        else
          v63 = *(_QWORD *)(v38 + 24);
        sub_13A3E40((__int64)&v65, (__int64)&v63);
        v68 = v58;
        if ( v58 > 0x40 )
          sub_16A4FD0(&v67, &v57);
        else
          v67 = v57;
        sub_16A9A30(&v69, &v67, &v65);
        sub_13A3610((__int64 *)&v57, (__int64 *)&v69);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( v68 > 0x40 && v67 )
          j_j___libc_free_0_0(v67);
        if ( v66 > 0x40 && v65 )
          j_j___libc_free_0_0(v65);
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
      }
      v34 = **(_QWORD **)(v34 + 32);
    }
    v41 = a3;
LABEL_137:
    while ( 2 )
    {
      if ( v39 <= 0x40 )
      {
        if ( v57 == (_QWORD *)1 )
          break;
        goto LABEL_130;
      }
      if ( v39 - (unsigned int)sub_16A57B0(&v57) > 0x40 || *v57 != 1 )
      {
LABEL_130:
        if ( *(_WORD *)(v41 + 24) != 7 )
          break;
        v42 = sub_13A5BC0((_QWORD *)v41, v35);
        if ( v52 == *(_QWORD ***)(v41 + 48) )
        {
          v49 = v42;
        }
        else
        {
          v43 = *(_WORD *)(v42 + 24);
          if ( v43 )
          {
            if ( v43 != 5 )
              goto LABEL_150;
            v42 = **(_QWORD **)(v42 + 32);
            if ( *(_WORD *)(v42 + 24) )
              goto LABEL_150;
          }
          sub_13A38D0((__int64)&v63, *(_QWORD *)(v42 + 32) + 24LL);
          sub_13A3E40((__int64)&v65, (__int64)&v63);
          v68 = v58;
          if ( v58 > 0x40 )
            sub_16A4FD0(&v67, &v57);
          else
            v67 = v57;
          sub_16A9A30(&v69, &v67, &v65);
          sub_13A3610((__int64 *)&v57, (__int64 *)&v69);
          sub_135E100((__int64 *)&v69);
          sub_135E100((__int64 *)&v67);
          sub_135E100((__int64 *)&v65);
          sub_135E100(&v63);
        }
        v39 = v58;
        v41 = **(_QWORD **)(v41 + 32);
        v35 = *(_QWORD *)(a1 + 8);
        continue;
      }
      break;
    }
    v44 = sub_14806B0(v35, v47, v49, 0, 0);
    v45 = *(_WORD *)(v44 + 24);
    if ( !v45 || v45 == 5 && (v44 = **(_QWORD **)(v44 + 32), !*(_WORD *)(v44 + 24)) )
    {
      sub_13A38D0((__int64)&v63, *(_QWORD *)(v44 + 32) + 24LL);
      sub_13A3E40((__int64)&v65, (__int64)&v63);
      sub_13A38D0((__int64)&v67, (__int64)&v57);
      sub_16A9A30(&v69, &v67, &v65);
      sub_13A3610((__int64 *)&v57, (__int64 *)&v69);
      sub_135E100((__int64 *)&v69);
      sub_135E100((__int64 *)&v67);
      sub_135E100((__int64 *)&v65);
      if ( !sub_13A38F0((__int64)&v57, 0) )
      {
        sub_16AB4D0(&v69, v61, &v57);
        sub_13A3610(v62, (__int64 *)&v69);
        sub_135E100((__int64 *)&v69);
        if ( !sub_13A38F0((__int64)v62, 0) )
        {
          v46 = (_BYTE *)(*(_QWORD *)(a4 + 48) + 16LL * ((unsigned int)sub_13A6B70(a1, v52) - 1));
          *v46 &= ~2u;
        }
      }
      sub_135E100(&v63);
    }
    if ( *(_WORD *)(v48 + 24) == 7 )
    {
      v33 = v48;
      continue;
    }
    break;
  }
LABEL_150:
  sub_135E100(v62);
LABEL_57:
  sub_135E100(v61);
LABEL_93:
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
LABEL_96:
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  return v53;
}
