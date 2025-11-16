// Function: sub_12407A0
// Address: 0x12407a0
//
__int64 __fastcall sub_12407A0(__int64 a1)
{
  unsigned __int64 v2; // rsi
  unsigned int v3; // r13d
  void *v4; // r12
  int v6; // eax
  int v7; // ecx
  __int64 v8; // r8
  int v9; // ecx
  unsigned int v10; // edi
  int *v11; // rdx
  int v12; // r10d
  __int64 v13; // r14
  unsigned int v14; // eax
  const char *v15; // rax
  void **j; // r14
  void **i; // r15
  const char *v18; // rax
  _BYTE *v19; // rax
  unsigned __int64 v20; // [rsp+0h] [rbp-230h]
  __int64 v21; // [rsp+8h] [rbp-228h]
  _DWORD *v22; // [rsp+38h] [rbp-1F8h]
  _QWORD v23[4]; // [rsp+40h] [rbp-1F0h] BYREF
  char v24; // [rsp+60h] [rbp-1D0h]
  char v25; // [rsp+61h] [rbp-1CFh]
  _BYTE *v26; // [rsp+70h] [rbp-1C0h] BYREF
  __int64 v27; // [rsp+78h] [rbp-1B8h]
  _BYTE v28[64]; // [rsp+80h] [rbp-1B0h] BYREF
  int v29; // [rsp+C0h] [rbp-170h] BYREF
  unsigned __int64 v30; // [rsp+C8h] [rbp-168h]
  int v31; // [rsp+D0h] [rbp-160h]
  __int64 v32; // [rsp+D8h] [rbp-158h]
  _QWORD *v33; // [rsp+E0h] [rbp-150h]
  unsigned __int64 v34; // [rsp+E8h] [rbp-148h]
  _QWORD v35[2]; // [rsp+F0h] [rbp-140h] BYREF
  _QWORD *v36; // [rsp+100h] [rbp-130h]
  __int64 v37; // [rsp+108h] [rbp-128h]
  _QWORD v38[2]; // [rsp+110h] [rbp-120h] BYREF
  __int64 v39; // [rsp+120h] [rbp-110h]
  unsigned int v40; // [rsp+128h] [rbp-108h]
  char v41; // [rsp+12Ch] [rbp-104h]
  void *v42; // [rsp+130h] [rbp-100h] BYREF
  void **v43; // [rsp+138h] [rbp-F8h]
  __int64 v44; // [rsp+150h] [rbp-E0h]
  char v45; // [rsp+158h] [rbp-D8h]
  __int64 v46; // [rsp+160h] [rbp-D0h] BYREF
  unsigned __int64 v47; // [rsp+168h] [rbp-C8h]
  __int64 v48; // [rsp+178h] [rbp-B8h]
  _QWORD *v49; // [rsp+180h] [rbp-B0h]
  unsigned __int64 v50; // [rsp+188h] [rbp-A8h]
  _QWORD v51[2]; // [rsp+190h] [rbp-A0h] BYREF
  _QWORD *v52; // [rsp+1A0h] [rbp-90h]
  __int64 v53; // [rsp+1A8h] [rbp-88h]
  _QWORD v54[2]; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v55; // [rsp+1C0h] [rbp-70h]
  unsigned int v56; // [rsp+1C8h] [rbp-68h]
  char v57; // [rsp+1CCh] [rbp-64h]
  void *v58; // [rsp+1D0h] [rbp-60h] BYREF
  void **v59; // [rsp+1D8h] [rbp-58h]
  __int64 v60; // [rsp+1F0h] [rbp-40h]
  char v61; // [rsp+1F8h] [rbp-38h]

  v20 = *(_QWORD *)(a1 + 232);
  v21 = a1 + 176;
  v29 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v33 = v35;
  v30 = 0;
  v32 = 0;
  v34 = 0;
  LOBYTE(v35[0]) = 0;
  v36 = v38;
  v37 = 0;
  LOBYTE(v38[0]) = 0;
  v40 = 1;
  v39 = 0;
  v41 = 0;
  v22 = sub_C33320();
  sub_C3B1B0((__int64)&v46, 0.0);
  sub_C407B0(&v42, &v46, v22);
  sub_C338F0((__int64)&v46);
  LOBYTE(v54[0]) = 0;
  v57 = 0;
  v49 = v51;
  v52 = v54;
  v44 = 0;
  v45 = 0;
  LODWORD(v46) = 0;
  v47 = 0;
  v48 = 0;
  v50 = 0;
  LOBYTE(v51[0]) = 0;
  v53 = 0;
  v56 = 1;
  v55 = 0;
  sub_C3B1B0((__int64)&v26, 0.0);
  sub_C407B0(&v58, (__int64 *)&v26, v22);
  sub_C338F0((__int64)&v26);
  v60 = 0;
  v2 = (unsigned __int64)&v29;
  v26 = v28;
  v61 = 0;
  v27 = 0x1000000000LL;
  if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)&v29, 0, 0) )
    goto LABEL_2;
  v2 = 4;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected comma in uselistorder_bb directive") )
    goto LABEL_2;
  v2 = (unsigned __int64)&v46;
  if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)&v46, 0, 0) )
    goto LABEL_2;
  v2 = 4;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected comma in uselistorder_bb directive") )
    goto LABEL_2;
  v2 = (unsigned __int64)&v26;
  if ( sub_1210710(a1, (__int64)&v26) )
    goto LABEL_2;
  v6 = v29;
  if ( v29 == 3 )
  {
    v13 = sub_BA8B30(*(_QWORD *)(a1 + 344), (__int64)v33, v34);
  }
  else
  {
    if ( v29 != 1 )
    {
LABEL_58:
      v25 = 1;
      v18 = "expected function name in uselistorder_bb";
LABEL_59:
      v2 = v30;
      v23[0] = v18;
      v3 = 1;
      v24 = 3;
      sub_11FD800(v21, v30, (__int64)v23, 1);
      goto LABEL_3;
    }
    v7 = *(_DWORD *)(a1 + 1216);
    v8 = *(_QWORD *)(a1 + 1200);
    if ( !v7 )
    {
LABEL_60:
      v25 = 1;
      v18 = "invalid function forward reference in uselistorder_bb";
      goto LABEL_59;
    }
    v9 = v7 - 1;
    v10 = v9 & (37 * v31);
    v11 = (int *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v31 != *v11 )
    {
      while ( v12 != -1 )
      {
        v10 = v9 & (v6 + v10);
        v11 = (int *)(v8 + 16LL * v10);
        v12 = *v11;
        if ( v31 == *v11 )
          goto LABEL_36;
        ++v6;
      }
      goto LABEL_60;
    }
LABEL_36:
    v13 = *((_QWORD *)v11 + 1);
  }
  if ( !v13 )
    goto LABEL_60;
  if ( *(_BYTE *)v13 )
    goto LABEL_58;
  LOBYTE(v14) = sub_B2FC80(v13);
  v3 = v14;
  if ( (_BYTE)v14 )
  {
    v2 = v30;
    v25 = 1;
    v23[0] = "invalid declaration in uselistorder_bb";
    v24 = 3;
    sub_11FD800(v21, v30, (__int64)v23, 1);
  }
  else
  {
    if ( !(_DWORD)v46 )
    {
      v25 = 1;
      v15 = "invalid numeric label in uselistorder_bb";
      goto LABEL_43;
    }
    if ( (_DWORD)v46 != 2 )
    {
      v25 = 1;
      v15 = "expected basic block name in uselistorder_bb";
LABEL_43:
      v2 = v47;
      v23[0] = v15;
      v24 = 3;
      sub_11FD800(v21, v47, (__int64)v23, 1);
LABEL_2:
      v3 = 1;
      goto LABEL_3;
    }
    v19 = (_BYTE *)sub_1209B90(*(_QWORD *)(v13 + 112), v49, v50);
    v2 = (unsigned __int64)v19;
    if ( !v19 )
    {
      v25 = 1;
      v15 = "invalid basic block in uselistorder_bb";
      goto LABEL_43;
    }
    if ( *v19 != 23 )
    {
      v25 = 1;
      v15 = "expected basic block in uselistorder_bb";
      goto LABEL_43;
    }
    v3 = sub_123FE40(a1, (__int64)v19, (__int64)v26, (unsigned int)v27, v20);
  }
LABEL_3:
  if ( v26 != v28 )
    _libc_free(v26, v2);
  if ( v60 )
    j_j___libc_free_0_0(v60);
  v4 = sub_C33340();
  if ( v58 == v4 )
  {
    if ( v59 )
    {
      for ( i = &v59[3 * (_QWORD)*(v59 - 1)]; v59 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v4 == *i )
            break;
          sub_C338F0((__int64)i);
          if ( v59 == i )
            goto LABEL_53;
        }
      }
LABEL_53:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v58);
  }
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v52 != v54 )
    j_j___libc_free_0(v52, v54[0] + 1LL);
  if ( v49 != v51 )
    j_j___libc_free_0(v49, v51[0] + 1LL);
  if ( v44 )
    j_j___libc_free_0_0(v44);
  if ( v4 == v42 )
  {
    if ( v43 )
    {
      for ( j = &v43[3 * (_QWORD)*(v43 - 1)]; v43 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v4 == *j )
            break;
          sub_C338F0((__int64)j);
          if ( v43 == j )
            goto LABEL_46;
        }
      }
LABEL_46:
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v42);
  }
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v36 != v38 )
    j_j___libc_free_0(v36, v38[0] + 1LL);
  if ( v33 != v35 )
    j_j___libc_free_0(v33, v35[0] + 1LL);
  return v3;
}
