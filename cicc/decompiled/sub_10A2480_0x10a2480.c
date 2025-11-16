// Function: sub_10A2480
// Address: 0x10a2480
//
unsigned __int8 *__fastcall sub_10A2480(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, bool a4)
{
  unsigned __int8 v6; // al
  unsigned __int8 *v7; // r14
  unsigned __int8 **v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  const void *v13; // rdx
  __int64 v14; // r14
  bool v15; // bl
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  bool v21; // r15
  __int64 v22; // rax
  char v23; // si
  unsigned __int64 v24; // r13
  char v25; // al
  int v26; // eax
  char v27; // r13
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdx
  _BYTE *v31; // rax
  __int64 v32; // rdx
  int v33; // eax
  unsigned __int64 v34; // rdx
  __int64 v35; // rcx
  int v36; // eax
  const void **v37; // [rsp+10h] [rbp-90h] BYREF
  __int64 v38; // [rsp+18h] [rbp-88h] BYREF
  __int64 v39; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-78h]
  const void *v41; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+38h] [rbp-68h]
  const void *v43; // [rsp+40h] [rbp-60h] BYREF
  const void ***v44; // [rsp+48h] [rbp-58h] BYREF
  char v45; // [rsp+50h] [rbp-50h]
  __int64 *v46; // [rsp+58h] [rbp-48h]
  __int16 v47; // [rsp+60h] [rbp-40h]

  v6 = *a2;
  if ( *a2 == 44 )
  {
    v17 = *((_QWORD *)a2 - 8);
    if ( !v17 )
      return 0;
    v18 = *((_QWORD *)a2 - 4);
    if ( !v18 )
      return 0;
    if ( *a3 != 44 )
      return 0;
    v19 = *((_QWORD *)a3 - 8);
    if ( !v19 || v17 != *((_QWORD *)a3 - 4) )
      return 0;
    v21 = a4;
    v47 = 257;
    v22 = sub_B504D0(15, v19, v18, (__int64)&v43, 0, 0);
    v23 = 0;
    v7 = (unsigned __int8 *)v22;
    if ( !a4 )
      goto LABEL_68;
    v24 = *a2;
    if ( (unsigned __int8)v24 <= 0x1Cu )
    {
      if ( (_BYTE)v24 != 5
        || (v33 = *((unsigned __int16 *)a2 + 1), (*((_WORD *)a2 + 1) & 0xFFFD) != 0xD) && (v33 & 0xFFF7) != 0x11 )
      {
        if ( !(unsigned __int8)sub_987880(a2) )
        {
LABEL_87:
          v27 = 0;
          v23 = 0;
          goto LABEL_50;
        }
        v21 = 0;
LABEL_70:
        v26 = *((unsigned __int16 *)a2 + 1);
        v23 = v21;
LABEL_43:
        v27 = 0;
        if ( v26 == 15 && (a2[1] & 2) != 0 && (unsigned __int8)sub_987880(a3) )
        {
          v28 = *a3;
          v29 = (unsigned __int8)v28 <= 0x1Cu ? *((unsigned __int16 *)a3 + 1) : v28 - 29;
          v27 = 0;
          if ( v29 == 15 )
            v27 = (a3[1] & 2) != 0;
        }
        goto LABEL_50;
      }
    }
    else
    {
      if ( (unsigned __int8)v24 > 0x36u )
      {
        v25 = sub_987880(a2);
        goto LABEL_40;
      }
      v25 = sub_987880(a2);
      v32 = 0x40540000000000LL;
      if ( !_bittest64(&v32, v24) )
      {
LABEL_40:
        if ( v25 )
        {
          v23 = 0;
LABEL_42:
          v26 = (unsigned __int8)v24 - 29;
          goto LABEL_43;
        }
        goto LABEL_87;
      }
      v33 = (unsigned __int8)v24 - 29;
    }
    if ( v33 != 15 || (a2[1] & 4) == 0 )
      goto LABEL_67;
    v34 = *a3;
    if ( (unsigned __int8)v34 <= 0x1Cu )
    {
      v23 = 0;
      v21 = 0;
      if ( (_BYTE)v34 != 5 )
        goto LABEL_68;
      v36 = *((unsigned __int16 *)a3 + 1);
      v21 = (*((_WORD *)a3 + 1) & 0xFFF7) == 17 || (*((_WORD *)a3 + 1) & 0xFFFD) == 13;
      if ( !v21 )
        goto LABEL_68;
    }
    else
    {
      if ( (unsigned __int8)v34 > 0x36u )
        goto LABEL_67;
      v35 = 0x40540000000000LL;
      if ( !_bittest64(&v35, v34) )
        goto LABEL_67;
      v36 = (unsigned __int8)v34 - 29;
    }
    if ( v36 == 15 )
    {
      v23 = (a3[1] & 4) != 0;
      v21 = v23;
      goto LABEL_68;
    }
LABEL_67:
    v23 = 0;
    v21 = 0;
LABEL_68:
    v27 = 0;
    if ( !(unsigned __int8)sub_987880(a2) )
    {
LABEL_50:
      sub_B44850(v7, v23);
      sub_B447F0(v7, v27);
      return v7;
    }
    LOBYTE(v24) = *a2;
    if ( *a2 > 0x1Cu )
      goto LABEL_42;
    goto LABEL_70;
  }
  v43 = a3;
  v44 = &v37;
  v45 = 0;
  v46 = &v38;
  LOBYTE(v47) = 0;
  if ( v6 != 54 )
    return 0;
  v9 = (unsigned __int8 **)*((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v9 != 49 || a3 != *(v9 - 8) || !(unsigned __int8)sub_991580((__int64)&v44, (__int64)*(v9 - 4)) )
    return 0;
  v10 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    *v46 = v10 + 24;
    goto LABEL_10;
  }
  v30 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
  if ( (unsigned int)v30 > 1 )
    return 0;
  if ( *(_BYTE *)v10 > 0x15u )
    return 0;
  v31 = sub_AD7630(v10, (unsigned __int8)v47, v30);
  if ( !v31 || *v31 != 17 )
    return 0;
  *v46 = (__int64)(v31 + 24);
LABEL_10:
  v40 = *(_DWORD *)(v38 + 8);
  if ( v40 > 0x40 )
    sub_C43690((__int64)&v39, 1, 0);
  else
    v39 = 1;
  v11 = *((_DWORD *)v37 + 2);
  LODWORD(v44) = v11;
  if ( v11 > 0x40 )
  {
    sub_C43780((__int64)&v43, v37);
    v11 = (unsigned int)v44;
    if ( (unsigned int)v44 > 0x40 )
    {
      sub_C43D10((__int64)&v43);
      goto LABEL_17;
    }
    v12 = (unsigned __int64)v43;
  }
  else
  {
    v12 = (unsigned __int64)*v37;
  }
  v13 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~v12);
  if ( !v11 )
    v13 = 0;
  v43 = v13;
LABEL_17:
  sub_C46250((__int64)&v43);
  v14 = v38;
  v42 = (unsigned int)v44;
  v41 = v43;
  LODWORD(v44) = v40;
  if ( v40 > 0x40 )
    sub_C43780((__int64)&v43, (const void **)&v39);
  else
    v43 = (const void *)v39;
  sub_C47AC0((__int64)&v43, v14);
  if ( v42 <= 0x40 )
    v15 = v41 == v43;
  else
    v15 = sub_C43C50((__int64)&v41, &v43);
  if ( (unsigned int)v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( !v15 )
  {
    if ( v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    return 0;
  }
  v16 = sub_AD8D80(*((_QWORD *)a3 + 1), (__int64)&v41);
  v47 = 257;
  v7 = (unsigned __int8 *)sub_B504D0(23, (__int64)a3, v16, (__int64)&v43, 0, 0);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  return v7;
}
