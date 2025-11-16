// Function: sub_1662A80
// Address: 0x1662a80
//
unsigned __int64 __fastcall sub_1662A80(__int64 **a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // ebx
  int v8; // r15d
  __int64 v9; // rax
  char v10; // r12
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 *v14; // r13
  const char *v15; // rax
  __int64 v16; // r12
  _BYTE *v17; // rax
  bool v18; // zf
  __int64 v19; // r10
  int v20; // eax
  unsigned int v21; // ecx
  __int64 v22; // rdx
  __int64 *v23; // rdi
  __int64 *v24; // r12
  char v25; // r9
  __int64 v26; // r8
  _BYTE *v27; // rax
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rdi
  _BYTE *v31; // rax
  unsigned __int64 result; // rax
  __int64 *v33; // rbx
  const char *v34; // rax
  __int64 v35; // r12
  _BYTE *v36; // rax
  unsigned __int8 *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // r12
  _BYTE *v44; // rax
  __int64 *v45; // rdi
  __int64 v46; // rdx
  __int64 *v47; // rdi
  __int64 v48; // [rsp+8h] [rbp-B8h]
  __int64 v49; // [rsp+8h] [rbp-B8h]
  char v50; // [rsp+10h] [rbp-B0h]
  char v51; // [rsp+10h] [rbp-B0h]
  char v52; // [rsp+10h] [rbp-B0h]
  char v53; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  __int64 v58; // [rsp+30h] [rbp-90h]
  unsigned __int64 v59; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v60[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v61; // [rsp+48h] [rbp-78h] BYREF
  unsigned __int64 v62; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-68h]
  char v64; // [rsp+60h] [rbp-60h]
  _QWORD v65[2]; // [rsp+70h] [rbp-50h] BYREF
  char v66; // [rsp+80h] [rbp-40h]
  char v67; // [rsp+81h] [rbp-3Fh]

  v4 = *(unsigned int *)(a3 + 8);
  v5 = (unsigned __int8 *)a3;
  v60[0] = (unsigned __int8 *)a3;
  if ( (_DWORD)v4 == 2 )
  {
    if ( (unsigned __int8)sub_1662870((__int64)a1, a3) )
      return 0;
    return 0xFFFFFFFF00000001LL;
  }
  if ( !a4 )
  {
    v7 = v4 & 1;
    if ( (v4 & 1) != 0 )
    {
      if ( !**(_BYTE **)(a3 - 8 * v4) )
      {
        v64 = 0;
        v8 = 2;
        goto LABEL_11;
      }
      v33 = *a1;
      if ( *a1 )
      {
        v67 = 1;
        v65[0] = "Struct tag nodes have a string as their first operand";
        v66 = 3;
        v43 = *v33;
        if ( *v33 )
        {
          sub_16E2CE0(v65, *v33);
          v44 = *(_BYTE **)(v43 + 24);
          if ( (unsigned __int64)v44 >= *(_QWORD *)(v43 + 16) )
          {
            sub_16E7DE0(v43, 10);
          }
          else
          {
            *(_QWORD *)(v43 + 24) = v44 + 1;
            *v44 = 10;
          }
          v40 = *v33;
          *((_BYTE *)v33 + 72) = 1;
          if ( !v40 )
            return 0xFFFFFFFF00000001LL;
          v37 = v60[0];
          if ( !v60[0] )
            return 0xFFFFFFFF00000001LL;
          v38 = v33[1];
          v39 = (__int64)(v33 + 2);
          goto LABEL_81;
        }
        *((_BYTE *)v33 + 72) = 1;
      }
    }
    else
    {
      v33 = *a1;
      if ( *a1 )
      {
        v67 = 1;
        v34 = "Struct tag nodes must have an odd number of operands!";
        goto LABEL_74;
      }
    }
    return 0xFFFFFFFF00000001LL;
  }
  if ( (unsigned int)(-1431655765 * v4) > 0x55555555 )
  {
    v33 = *a1;
    if ( *a1 )
    {
      v67 = 1;
      v34 = "Access tag nodes must have the number of operands that is a multiple of 3!";
LABEL_74:
      v65[0] = v34;
      v66 = 3;
      v35 = *v33;
      if ( *v33 )
      {
        sub_16E2CE0(v65, *v33);
        v36 = *(_BYTE **)(v35 + 24);
        if ( (unsigned __int64)v36 >= *(_QWORD *)(v35 + 16) )
        {
          sub_16E7DE0(v35, 10);
        }
        else
        {
          *(_QWORD *)(v35 + 24) = v36 + 1;
          *v36 = 10;
        }
        v35 = *v33;
      }
      *((_BYTE *)v33 + 72) = 1;
      if ( !v35 )
        return 0xFFFFFFFF00000001LL;
      v37 = v60[0];
      if ( !v60[0] )
        return 0xFFFFFFFF00000001LL;
      v38 = v33[1];
      v39 = (__int64)(v33 + 2);
      v40 = v35;
LABEL_81:
      sub_15562E0(v37, v40, v39, v38);
      v41 = *v33;
      v42 = *(_BYTE **)(*v33 + 24);
      if ( (unsigned __int64)v42 >= *(_QWORD *)(*v33 + 16) )
      {
        sub_16E7DE0(v41, 10);
      }
      else
      {
        *(_QWORD *)(v41 + 24) = v42 + 1;
        *v42 = 10;
      }
      return 0xFFFFFFFF00000001LL;
    }
    return 0xFFFFFFFF00000001LL;
  }
  v6 = *(_QWORD *)(a3 + 8 * (1 - v4));
  if ( !v6 || *(_BYTE *)v6 != 1 || *(_BYTE *)(*(_QWORD *)(v6 + 136) + 16LL) != 13 )
  {
    v47 = *a1;
    v62 = a2;
    if ( v47 )
    {
      v67 = 1;
      v65[0] = "Type size nodes must be constants!";
      v66 = 3;
      sub_165C320(v47, (__int64)v65, (__int64 *)&v62, v60);
    }
    return 0xFFFFFFFF00000001LL;
  }
  v64 = 0;
  v7 = 3;
  v8 = 3;
LABEL_11:
  LODWORD(v58) = -1;
  v9 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v9 <= v7 )
    goto LABEL_66;
  v10 = 0;
  do
  {
    if ( (unsigned __int8)(**(_BYTE **)&v5[8 * (v7 - v9)] - 4) > 0x1Eu )
    {
      v14 = *a1;
      if ( !*a1 )
        goto LABEL_27;
      v67 = 1;
      v15 = "Incorrect field entry in struct type node!";
      goto LABEL_20;
    }
    v12 = *(_QWORD *)&v5[8 * (v7 + 1 - v9)];
    if ( !v12 || *(_BYTE *)v12 != 1 || (v13 = *(_QWORD *)(v12 + 136), *(_BYTE *)(v13 + 16) != 13) )
    {
      v45 = *a1;
      v61 = a2;
      if ( v45 )
      {
        v67 = 1;
        v65[0] = "Offset entries must be constants!";
        v66 = 3;
        sub_165C320(v45, (__int64)v65, &v61, v60);
        v5 = v60[0];
      }
      goto LABEL_27;
    }
    if ( (_DWORD)v58 == -1 )
    {
      LODWORD(v58) = *(_DWORD *)(v13 + 32);
    }
    else if ( (_DWORD)v58 != *(_DWORD *)(v13 + 32) )
    {
      v14 = *a1;
      if ( !*a1 )
        goto LABEL_27;
      v67 = 1;
      v15 = "Bitwidth between the offsets and struct type entries must match";
LABEL_20:
      v65[0] = v15;
      v66 = 3;
      v16 = *v14;
      if ( *v14 )
      {
        sub_16E2CE0(v65, *v14);
        v17 = *(_BYTE **)(v16 + 24);
        if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 16) )
        {
          sub_16E7DE0(v16, 10);
        }
        else
        {
          *(_QWORD *)(v16 + 24) = v17 + 1;
          *v17 = 10;
        }
      }
      v18 = *v14 == 0;
      *((_BYTE *)v14 + 72) = 1;
      if ( v18 )
        goto LABEL_26;
      sub_164FA80(v14, a2);
      v5 = v60[0];
      if ( v60[0] )
      {
        sub_164ED40(v14, v60[0]);
LABEL_26:
        v5 = v60[0];
      }
LABEL_27:
      v9 = *((unsigned int *)v5 + 2);
      v10 = 1;
      goto LABEL_28;
    }
    v19 = v13 + 24;
    v53 = v64;
    if ( v64 )
    {
      v20 = sub_16A9900(&v62, v13 + 24);
      v19 = v13 + 24;
      if ( v20 <= 0 )
        goto LABEL_35;
      v24 = *a1;
      v25 = v53;
      if ( !*a1 )
      {
        v10 = v53;
LABEL_35:
        if ( v63 <= 0x40 && (v21 = *(_DWORD *)(v13 + 32), v21 <= 0x40) )
        {
          v46 = *(_QWORD *)(v13 + 24);
          v63 = *(_DWORD *)(v13 + 32);
          v62 = v46 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v21);
        }
        else
        {
          sub_16A51C0(&v62, v19);
        }
        goto LABEL_38;
      }
      v67 = 1;
      v65[0] = "Offsets must be increasing!";
      v66 = 3;
      v26 = *v24;
      if ( *v24 )
      {
        v50 = v53;
        v54 = *v24;
        sub_16E2CE0(v65, *v24);
        v25 = v50;
        v19 = v13 + 24;
        v27 = *(_BYTE **)(v54 + 24);
        if ( (unsigned __int64)v27 >= *(_QWORD *)(v54 + 16) )
        {
          sub_16E7DE0(v54, 10);
          v26 = *v24;
          v25 = v50;
          v19 = v13 + 24;
        }
        else
        {
          *(_QWORD *)(v54 + 24) = v27 + 1;
          *v27 = 10;
          v26 = *v24;
        }
      }
      *((_BYTE *)v24 + 72) = 1;
      if ( v26 )
      {
        v55 = (__int64)(v24 + 2);
        v48 = v19;
        v51 = v25;
        if ( *(_BYTE *)(a2 + 16) > 0x17u )
          sub_155BD40(a2, v26, (__int64)(v24 + 2), 0);
        else
          sub_1553920((__int64 *)a2, v26, 1, v55);
        v28 = *v24;
        v25 = v51;
        v19 = v48;
        v29 = *(_BYTE **)(*v24 + 24);
        if ( (unsigned __int64)v29 >= *(_QWORD *)(*v24 + 16) )
        {
          sub_16E7DE0(v28, 10);
          v25 = v51;
          v19 = v48;
        }
        else
        {
          *(_QWORD *)(v28 + 24) = v29 + 1;
          *v29 = 10;
        }
        if ( v60[0] )
        {
          v49 = v19;
          v52 = v25;
          sub_15562E0(v60[0], *v24, v55, v24[1]);
          v30 = *v24;
          v25 = v52;
          v19 = v49;
          v31 = *(_BYTE **)(*v24 + 24);
          if ( (unsigned __int64)v31 >= *(_QWORD *)(*v24 + 16) )
          {
            sub_16E7DE0(v30, 10);
            v25 = v52;
            v19 = v49;
          }
          else
          {
            *(_QWORD *)(v30 + 24) = v31 + 1;
            *v31 = 10;
          }
        }
      }
      v10 = v64;
      if ( v64 )
        goto LABEL_35;
      v10 = v25;
    }
    v63 = *(_DWORD *)(v13 + 32);
    if ( v63 > 0x40 )
      sub_16A4FD0(&v62, v19);
    else
      v62 = *(_QWORD *)(v13 + 24);
    v64 = 1;
LABEL_38:
    v5 = v60[0];
    v9 = *((unsigned int *)v60[0] + 2);
    if ( a4 )
    {
      v22 = *(_QWORD *)&v60[0][8 * (v7 + 2 - (unsigned __int64)(unsigned int)v9)];
      if ( !v22 || *(_BYTE *)v22 != 1 || *(_BYTE *)(*(_QWORD *)(v22 + 136) + 16LL) != 13 )
      {
        v23 = *a1;
        v61 = a2;
        if ( v23 )
        {
          v67 = 1;
          v65[0] = "Member size entries must be constants!";
          v66 = 3;
          sub_165C320(v23, (__int64)v65, &v61, v60);
          v5 = v60[0];
          v9 = *((unsigned int *)v60[0] + 2);
        }
        v10 = a4;
      }
    }
LABEL_28:
    v7 += v8;
  }
  while ( v7 < (unsigned int)v9 );
  result = 0xFFFFFFFF00000001LL;
  if ( !v10 )
  {
LABEL_66:
    result = v58 << 32;
    if ( v64 )
      goto LABEL_67;
    return result;
  }
  if ( !v64 )
    return result;
LABEL_67:
  if ( v63 > 0x40 )
  {
    if ( v62 )
    {
      v59 = result;
      j_j___libc_free_0_0(v62);
      return v59;
    }
  }
  return result;
}
