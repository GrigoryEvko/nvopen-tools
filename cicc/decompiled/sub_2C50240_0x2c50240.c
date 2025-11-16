// Function: sub_2C50240
// Address: 0x2c50240
//
__int64 __fastcall sub_2C50240(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rax
  char v14; // bl
  __int64 v15; // [rsp+0h] [rbp-110h]
  unsigned __int64 v19; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v21; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v23; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-B8h]
  unsigned __int64 v25; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v26; // [rsp+68h] [rbp-A8h]
  unsigned __int64 v27; // [rsp+70h] [rbp-A0h]
  unsigned int v28; // [rsp+78h] [rbp-98h]
  unsigned __int64 v29; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-88h]
  unsigned __int64 v31; // [rsp+90h] [rbp-80h]
  unsigned int v32; // [rsp+98h] [rbp-78h]
  unsigned __int64 v33; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v35; // [rsp+B0h] [rbp-60h]
  unsigned int v36; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v37; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v38; // [rsp+C8h] [rbp-48h]
  unsigned __int64 v39; // [rsp+D0h] [rbp-40h]
  unsigned int v40; // [rsp+D8h] [rbp-38h]

  if ( *(_BYTE *)a3 == 17 )
  {
    v7 = *(_DWORD *)(a3 + 32);
    if ( v7 > 0x40 )
    {
      if ( v7 - (unsigned int)sub_C444A0(a3 + 24) > 0x40 || (unsigned __int64)a2 <= **(_QWORD **)(a3 + 24) )
        goto LABEL_4;
    }
    else if ( (unsigned __int64)a2 <= *(_QWORD *)(a3 + 24) )
    {
LABEL_4:
      *(_DWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    *(_DWORD *)a1 = 1;
    *(_QWORD *)(a1 + 8) = 0;
    return a1;
  }
  v9 = sub_BCB060(*(_QWORD *)(a3 + 8));
  v20 = v9;
  v10 = v9;
  if ( v9 <= 0x40 )
  {
    v19 = 0;
    v22 = v9;
    v21 = a2;
    v38 = v9;
LABEL_11:
    v37 = v21;
    goto LABEL_12;
  }
  sub_C43690((__int64)&v19, 0, 0);
  v22 = v10;
  sub_C43690((__int64)&v21, a2, 0);
  v38 = v22;
  if ( v22 <= 0x40 )
    goto LABEL_11;
  sub_C43780((__int64)&v37, (const void **)&v21);
LABEL_12:
  v34 = v20;
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v33, (const void **)&v19);
  else
    v33 = v19;
  sub_AADC30((__int64)&v25, (__int64)&v33, (__int64 *)&v37);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  sub_AADB10((__int64)&v29, v10, 1);
  if ( sub_98ED70((unsigned __int8 *)a3, a5, 0, 0, 0) )
  {
    sub_99D930((__int64)&v37, (unsigned __int8 *)a3, 0, 1u, a5, a4, a6, 0);
    v14 = sub_AB1BB0((__int64)&v25, (__int64)&v37);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( !v14 )
      goto LABEL_24;
    *(_DWORD *)a1 = 1;
    *(_QWORD *)(a1 + 8) = 0;
    goto LABEL_25;
  }
  if ( *(_BYTE *)a3 == 57 )
  {
    v15 = *(_QWORD *)(a3 - 64);
    if ( v15 )
    {
      v11 = *(_QWORD *)(a3 - 32);
      if ( *(_BYTE *)v11 == 17 )
      {
        v24 = *(_DWORD *)(v11 + 32);
        if ( v24 > 0x40 )
          sub_C43780((__int64)&v23, (const void **)(v11 + 24));
        else
          v23 = *(_QWORD *)(v11 + 24);
        sub_AADBC0((__int64)&v33, (__int64 *)&v23);
        sub_AB8410((__int64)&v37, (__int64)&v29, (__int64)&v33);
        if ( v30 > 0x40 )
        {
LABEL_49:
          if ( v29 )
            j_j___libc_free_0_0(v29);
        }
LABEL_51:
        v29 = v37;
        v12 = v38;
        v38 = 0;
        v30 = v12;
        if ( v32 > 0x40 && v31 )
        {
          j_j___libc_free_0_0(v31);
          v31 = v39;
          v32 = v40;
          if ( v38 > 0x40 && v37 )
            j_j___libc_free_0_0(v37);
        }
        else
        {
          v31 = v39;
          v32 = v40;
        }
        if ( v36 > 0x40 && v35 )
          j_j___libc_free_0_0(v35);
        if ( v34 > 0x40 && v33 )
          j_j___libc_free_0_0(v33);
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
      }
    }
  }
  else if ( *(_BYTE *)a3 == 51 )
  {
    v15 = *(_QWORD *)(a3 - 64);
    if ( v15 )
    {
      v13 = *(_QWORD *)(a3 - 32);
      if ( *(_BYTE *)v13 == 17 )
      {
        v24 = *(_DWORD *)(v13 + 32);
        if ( v24 > 0x40 )
          sub_C43780((__int64)&v23, (const void **)(v13 + 24));
        else
          v23 = *(_QWORD *)(v13 + 24);
        sub_AADBC0((__int64)&v33, (__int64 *)&v23);
        sub_AB8080((__int64)&v37, (__int64)&v29, (__int64 *)&v33);
        if ( v30 > 0x40 )
          goto LABEL_49;
        goto LABEL_51;
      }
    }
  }
  if ( !(unsigned __int8)sub_AB1BB0((__int64)&v25, (__int64)&v29) )
  {
LABEL_24:
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    goto LABEL_25;
  }
  *(_DWORD *)a1 = 2;
  *(_QWORD *)(a1 + 8) = v15;
LABEL_25:
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
