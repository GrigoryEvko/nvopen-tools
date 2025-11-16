// Function: sub_1217520
// Address: 0x1217520
//
__int64 __fastcall sub_1217520(__int64 a1, _QWORD **a2)
{
  __int64 v2; // r13
  unsigned int v3; // r12d
  unsigned __int64 v4; // rsi
  __int64 *v5; // rbx
  __int64 *v6; // r13
  __int64 v7; // rdi
  const void **v9; // rsi
  unsigned int v10; // r15d
  const char *v11; // rax
  const char *v12; // rax
  const void **v13; // rsi
  unsigned int v14; // r15d
  const char *v15; // rax
  int v16; // eax
  bool v17; // cc
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  int v20; // ecx
  __int64 *v21; // rdx
  const char **v22; // rsi
  __int64 *v23; // rax
  int v24; // edx
  int v25; // edx
  const char *v26; // rbx
  const char *v27; // r13
  __int64 v28; // rdi
  const char *v29; // [rsp+8h] [rbp-158h]
  const char *v30; // [rsp+8h] [rbp-158h]
  __int64 *v31; // [rsp+8h] [rbp-158h]
  const char *v33; // [rsp+20h] [rbp-140h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-138h]
  const char *v35; // [rsp+30h] [rbp-130h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-128h]
  const char *v37; // [rsp+40h] [rbp-120h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-118h]
  const char *v39; // [rsp+50h] [rbp-110h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-108h]
  char v41; // [rsp+70h] [rbp-F0h]
  char v42; // [rsp+71h] [rbp-EFh]
  __int64 *v43; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v44; // [rsp+88h] [rbp-D8h]
  _BYTE v45[64]; // [rsp+90h] [rbp-D0h] BYREF
  const char *v46; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int v47; // [rsp+D8h] [rbp-88h]
  __int64 v48; // [rsp+E0h] [rbp-80h] BYREF
  unsigned int v49; // [rsp+E8h] [rbp-78h]
  char v50; // [rsp+F0h] [rbp-70h]
  char v51; // [rsp+F1h] [rbp-6Fh]
  char v52; // [rsp+120h] [rbp-40h]

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v3 = sub_120AFE0(a1, 12, "expected '('");
  if ( (_BYTE)v3 )
    return v3;
  v43 = (__int64 *)v45;
  v44 = 0x200000000LL;
  while ( 1 )
  {
    v4 = 12;
    v34 = 1;
    v33 = 0;
    v36 = 1;
    v35 = 0;
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '('") )
      goto LABEL_6;
    if ( *(_DWORD *)(a1 + 240) != 529 )
    {
      v4 = *(_QWORD *)(a1 + 232);
      v51 = 1;
      v46 = "expected integer";
      v50 = 3;
      sub_11FD800(v2, v4, (__int64)&v46, 1);
      goto LABEL_6;
    }
    v9 = (const void **)(a1 + 320);
    if ( *(_BYTE *)(a1 + 332) )
      sub_C449B0((__int64)&v46, v9, 0x40u);
    else
      sub_C44830((__int64)&v46, v9, 0x40u);
    v10 = v47;
    v11 = v46;
    if ( v34 > 0x40 && v33 )
    {
      v29 = v46;
      j_j___libc_free_0_0(v33);
      v11 = v29;
    }
    v33 = v11;
    v34 = v10;
    v4 = 4;
    *(_DWORD *)(a1 + 240) = sub_1205200(v2);
    if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ','") )
      goto LABEL_6;
    if ( *(_DWORD *)(a1 + 240) != 529 )
    {
      v51 = 1;
      v12 = "expected integer";
      goto LABEL_31;
    }
    v13 = (const void **)(a1 + 320);
    if ( *(_BYTE *)(a1 + 332) )
      sub_C449B0((__int64)&v46, v13, 0x40u);
    else
      sub_C44830((__int64)&v46, v13, 0x40u);
    v14 = v47;
    v15 = v46;
    if ( v36 > 0x40 && v35 )
    {
      v30 = v46;
      j_j___libc_free_0_0(v35);
      v15 = v30;
    }
    v35 = v15;
    v36 = v14;
    v16 = sub_1205200(v2);
    v17 = v34 <= 0x40;
    *(_DWORD *)(a1 + 240) = v16;
    if ( v17 )
    {
      if ( v33 == v35 )
      {
LABEL_86:
        v51 = 1;
        v12 = "the range should not represent the full or empty set!";
LABEL_31:
        v50 = 3;
        v4 = *(_QWORD *)(a1 + 232);
        v46 = v12;
        sub_11FD800(v2, v4, (__int64)&v46, 1);
LABEL_6:
        if ( v36 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
        if ( v34 > 0x40 && v33 )
          j_j___libc_free_0_0(v33);
        v3 = 1;
        goto LABEL_11;
      }
    }
    else if ( sub_C43C50((__int64)&v33, (const void **)&v35) )
    {
      goto LABEL_86;
    }
    v4 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')'") )
      goto LABEL_6;
    v40 = v36;
    if ( v36 > 0x40 )
      sub_C43780((__int64)&v39, (const void **)&v35);
    else
      v39 = v35;
    v38 = v34;
    if ( v34 > 0x40 )
      sub_C43780((__int64)&v37, (const void **)&v33);
    else
      v37 = v33;
    sub_AADC30((__int64)&v46, (__int64)&v37, (__int64 *)&v39);
    v18 = (unsigned int)v44;
    v19 = (unsigned int)v44 + 1LL;
    v20 = v44;
    if ( v19 > HIDWORD(v44) )
    {
      if ( v43 > (__int64 *)&v46 || (v31 = v43, &v46 >= (const char **)&v43[4 * (unsigned int)v44]) )
      {
        sub_9D5330((__int64)&v43, v19);
        v18 = (unsigned int)v44;
        v21 = v43;
        v22 = &v46;
        v20 = v44;
      }
      else
      {
        sub_9D5330((__int64)&v43, v19);
        v21 = v43;
        v18 = (unsigned int)v44;
        v22 = (const char **)((char *)v43 + (char *)&v46 - (char *)v31);
        v20 = v44;
      }
    }
    else
    {
      v21 = v43;
      v22 = &v46;
    }
    v23 = &v21[4 * v18];
    if ( v23 )
    {
      v24 = *((_DWORD *)v22 + 2);
      *((_DWORD *)v22 + 2) = 0;
      *((_DWORD *)v23 + 2) = v24;
      *v23 = (__int64)*v22;
      v25 = *((_DWORD *)v22 + 6);
      *((_DWORD *)v22 + 6) = 0;
      v20 = v44;
      *((_DWORD *)v23 + 6) = v25;
      v23[2] = (__int64)v22[2];
    }
    LODWORD(v44) = v20 + 1;
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(v2);
  }
  v4 = 13;
  v3 = sub_120AFE0(a1, 13, "expected ')'");
  if ( !(_BYTE)v3 )
  {
    sub_AC0280((__int64)&v46, v43, (unsigned int)v44);
    if ( v52 )
    {
      v4 = (unsigned __int64)&v46;
      sub_A790B0(a2, (__int64)&v46);
    }
    else
    {
      v4 = *(_QWORD *)(a1 + 232);
      v42 = 1;
      v3 = 1;
      v39 = "Invalid (unordered or overlapping) range list";
      v41 = 3;
      sub_11FD800(v2, v4, (__int64)&v39, 1);
    }
    if ( v52 )
    {
      v26 = v46;
      v52 = 0;
      v27 = &v46[32 * v47];
      if ( v46 != v27 )
      {
        do
        {
          v27 -= 32;
          if ( *((_DWORD *)v27 + 6) > 0x40u )
          {
            v28 = *((_QWORD *)v27 + 2);
            if ( v28 )
              j_j___libc_free_0_0(v28);
          }
          if ( *((_DWORD *)v27 + 2) > 0x40u && *(_QWORD *)v27 )
            j_j___libc_free_0_0(*(_QWORD *)v27);
        }
        while ( v26 != v27 );
        v27 = v46;
      }
      if ( v27 != (const char *)&v48 )
        _libc_free(v27, v4);
    }
  }
LABEL_11:
  v5 = v43;
  v6 = &v43[4 * (unsigned int)v44];
  if ( v43 != v6 )
  {
    do
    {
      v6 -= 4;
      if ( *((_DWORD *)v6 + 6) > 0x40u )
      {
        v7 = v6[2];
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
      if ( *((_DWORD *)v6 + 2) > 0x40u && *v6 )
        j_j___libc_free_0_0(*v6);
    }
    while ( v5 != v6 );
    v6 = v43;
  }
  if ( v6 != (__int64 *)v45 )
    _libc_free(v6, v4);
  return v3;
}
