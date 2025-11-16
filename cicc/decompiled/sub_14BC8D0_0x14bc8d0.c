// Function: sub_14BC8D0
// Address: 0x14bc8d0
//
bool *__fastcall sub_14BC8D0(
        bool *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 a7,
        int a8)
{
  int *v12; // rax
  int v13; // eax
  char v14; // al
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rsi
  char v20; // cl
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned int v24; // r14d
  char v25; // al
  __int64 v26; // rsi
  int v27; // eax
  unsigned int v28; // eax
  unsigned int v29; // eax
  bool v30; // bl
  bool v31; // r13
  __int64 v32; // [rsp+0h] [rbp-D0h]
  __int64 v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+8h] [rbp-C8h]
  __int64 v36; // [rsp+18h] [rbp-B8h]
  __int64 v37; // [rsp+18h] [rbp-B8h]
  __int64 v38; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+18h] [rbp-B8h]
  __int64 v40; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-A8h]
  __int64 v42; // [rsp+30h] [rbp-A0h]
  unsigned int v43; // [rsp+38h] [rbp-98h]
  __int64 v44; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-88h]
  __int64 v46; // [rsp+50h] [rbp-80h]
  unsigned int v47; // [rsp+58h] [rbp-78h]
  __int64 v48; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+68h] [rbp-68h]
  __int64 v50; // [rsp+70h] [rbp-60h]
  unsigned int v51; // [rsp+78h] [rbp-58h]
  __int64 v52; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v53; // [rsp+88h] [rbp-48h]
  __int64 v54; // [rsp+90h] [rbp-40h]
  unsigned int v55; // [rsp+98h] [rbp-38h]

  v12 = (int *)sub_16D40F0(qword_4FBB370);
  if ( v12 )
    v13 = *v12;
  else
    v13 = qword_4FBB370[2];
  if ( a8 == v13 )
    goto LABEL_5;
  v14 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( (*(_BYTE *)(*(_QWORD *)a4 + 8LL) == 16) != (v14 == 16) )
    goto LABEL_5;
  if ( v14 == 16 )
    goto LABEL_5;
  v16 = *(_BYTE *)(a2 + 16);
  if ( v16 <= 0x17u )
    goto LABEL_5;
  if ( v16 == 76 )
  {
    if ( a3 > 0xF || !byte_4F9D720 )
      goto LABEL_5;
    v22 = *(_QWORD *)(a2 - 48);
    v23 = *(_QWORD *)(a2 - 24);
    v24 = *(_WORD *)(a2 + 18) & 0x7FFF;
    if ( !a7 )
    {
      v34 = *(_QWORD *)(a2 - 24);
      v38 = *(_QWORD *)(a2 - 48);
      v29 = sub_15FF0F0(*(_WORD *)(a2 + 18) & 0x7FFF);
      v23 = v34;
      v22 = v38;
      v24 = v29;
    }
    v25 = a4 == v23 && a5 == v22;
    if ( (a4 != v22 || a5 != v23) && !v25 )
    {
      if ( a4 == v22 && *(_BYTE *)(v23 + 16) == 14 && *(_BYTE *)(a5 + 16) == 14 )
      {
        v27 = sub_14A9E40(v23 + 24, a5 + 24);
        if ( v24 - 4 <= 1 )
        {
          if ( (a3 - 4 <= 1 || a3 == 12) && !v27 )
            goto LABEL_41;
        }
        else if ( v24 - 2 <= 1 && a3 - 2 <= 1 && v27 == 2 )
        {
          goto LABEL_41;
        }
      }
      goto LABEL_5;
    }
    sub_14A8F90((__int64)&v52, v24, a3, v25);
    if ( !BYTE1(v52) )
      goto LABEL_5;
LABEL_16:
    v21 = v52;
    a1[1] = 1;
    *a1 = v21;
    return a1;
  }
  if ( v16 != 75 )
  {
    if ( (unsigned __int8)(v16 - 50) > 1u )
      goto LABEL_5;
    if ( a7 )
    {
      if ( v16 != 50 )
        goto LABEL_5;
    }
    else if ( v16 != 51 )
    {
      goto LABEL_5;
    }
    v26 = *(_QWORD *)(a2 - 48);
    if ( !v26 )
      goto LABEL_5;
    v32 = *(_QWORD *)(a2 - 24);
    if ( !v32 )
      goto LABEL_5;
    sub_14BC8D0((unsigned int)&v52, v26, a3, a4, a5, a6, a7, a8 + 1);
    if ( !BYTE1(v52) )
    {
      sub_14BC8D0((unsigned int)&v52, v32, a3, a4, a5, a6, a7, a8 + 1);
      if ( !BYTE1(v52) )
        goto LABEL_5;
    }
    goto LABEL_16;
  }
  v17 = *(_QWORD *)(a2 - 48);
  v18 = *(_QWORD *)(a2 - 24);
  v19 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( !a7 )
  {
    v33 = *(_QWORD *)(a2 - 24);
    v37 = v17;
    v28 = sub_15FF0F0((unsigned int)v19);
    v18 = v33;
    v17 = v37;
    v19 = v28;
  }
  v20 = a4 == v18 && a5 == v17;
  if ( a4 == v17 && a5 == v18 || v20 )
  {
    sub_14A8F90((__int64)&v52, v19, a3, v20);
    if ( !BYTE1(v52) )
      goto LABEL_5;
    goto LABEL_16;
  }
  if ( a4 == v17 && *(_BYTE *)(v18 + 16) == 13 && *(_BYTE *)(a5 + 16) == 13 )
  {
    sub_158B890(&v40, v19, v18 + 24);
    v49 = *(_DWORD *)(a5 + 32);
    if ( v49 > 0x40 )
      sub_16A4FD0(&v48, a5 + 24);
    else
      v48 = *(_QWORD *)(a5 + 24);
    sub_1589870(&v52, &v48);
    sub_158AE10(&v44, a3, &v52);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    sub_158BE00(&v48, &v40, &v44);
    sub_1590FF0(&v52, &v40, &v44);
    if ( (unsigned __int8)sub_158A120(&v48) )
    {
      v30 = 1;
      v31 = 0;
    }
    else
    {
      v30 = (unsigned __int8)sub_158A120(&v52) != 0;
      v31 = v30;
    }
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    if ( v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v30 )
    {
      a1[1] = 1;
      *a1 = v31;
      return a1;
    }
    goto LABEL_5;
  }
  if ( a3 == (_DWORD)v19 )
  {
    if ( a3 > 0x25 )
    {
      if ( a3 - 40 <= 1 )
      {
        v39 = v18;
        if ( (unsigned __int8)sub_14BC210(41, a4, v17, a6, a8) )
        {
          if ( (unsigned __int8)sub_14BC210(41, v39, a5, a6, a8) )
            goto LABEL_41;
        }
      }
    }
    else if ( a3 > 0x23 )
    {
      v36 = v18;
      if ( (unsigned __int8)sub_14BC210(37, a4, v17, a6, a8) )
      {
        if ( (unsigned __int8)sub_14BC210(37, v36, a5, a6, a8) )
        {
LABEL_41:
          *(_WORD *)a1 = 257;
          return a1;
        }
      }
    }
  }
LABEL_5:
  a1[1] = 0;
  return a1;
}
