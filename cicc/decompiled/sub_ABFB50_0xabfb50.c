// Function: sub_ABFB50
// Address: 0xabfb50
//
_QWORD *__fastcall sub_ABFB50(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rsi
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rsi
  unsigned __int64 *v10; // r13
  __int64 v11; // r12
  __int64 v12; // rsi
  unsigned int v13; // r14d
  __int64 v15; // rdx
  int v16; // esi
  unsigned __int64 v17; // rcx
  char *v18; // rax
  unsigned __int64 v19; // rdx
  int v20; // ecx
  int v21; // ecx
  int v22; // eax
  __int64 *v23; // r12
  __int64 *v24; // rbx
  __int64 v25; // rdi
  _QWORD *v26; // [rsp+10h] [rbp-110h]
  unsigned __int64 v27; // [rsp+18h] [rbp-108h]
  __int64 v29; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v31; // [rsp+38h] [rbp-E8h]
  __int64 v32; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-D8h]
  __int64 v34; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v35; // [rsp+58h] [rbp-C8h]
  __int64 v36; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v37; // [rsp+68h] [rbp-B8h]
  __int64 v38; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-A8h]
  __int64 v40; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v41; // [rsp+88h] [rbp-98h]
  __int64 v42; // [rsp+90h] [rbp-90h]
  unsigned int v43; // [rsp+98h] [rbp-88h]
  __int64 *v44; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v45; // [rsp+A8h] [rbp-78h]
  _BYTE v46[112]; // [rsp+B0h] [rbp-70h] BYREF

  v4 = (unsigned __int64 *)*(unsigned int *)(a2 + 8);
  v26 = a1 + 2;
  if ( !(_DWORD)v4 )
  {
    *a1 = v26;
    a1[1] = 0x200000000LL;
    if ( *(_DWORD *)(a2 + 8) )
      sub_ABF850((__int64)a1, (__int64 *)a2);
    return a1;
  }
  v5 = *(unsigned int *)(a3 + 8);
  if ( !(_DWORD)v5 )
  {
    *a1 = v26;
    a1[1] = 0x200000000LL;
    if ( *(_DWORD *)(a3 + 8) )
      sub_ABF850((__int64)a1, (__int64 *)a3);
    return a1;
  }
  v29 = a2;
  v6 = 0;
  v44 = (__int64 *)v46;
  v45 = 0x200000000LL;
  v31 = 0;
  while ( v5 > v6 )
  {
    v7 = *(_QWORD *)v29 + 32 * v31;
    v8 = *(_QWORD *)a3 + 32 * v6;
    v9 = v8;
    if ( (int)sub_C4C880(v7, v8) > 0 )
      v9 = v7;
    v33 = *(_DWORD *)(v9 + 8);
    if ( v33 > 0x40 )
      sub_C43780(&v32, v9);
    else
      v32 = *(_QWORD *)v9;
    v10 = (unsigned __int64 *)(v8 + 16);
    v11 = v7 + 16;
    v12 = (__int64)v10;
    if ( (int)sub_C4C880(v11, v10) < 0 )
      v12 = v11;
    v13 = *(_DWORD *)(v12 + 8);
    v35 = v13;
    if ( v13 <= 0x40 )
    {
      v34 = *(_QWORD *)v12;
      if ( (int)sub_C4C880(&v32, &v34) >= 0 )
        goto LABEL_22;
      v39 = v13;
LABEL_51:
      v38 = v34;
      goto LABEL_30;
    }
    sub_C43780(&v34, v12);
    v13 = v35;
    if ( (int)sub_C4C880(&v32, &v34) >= 0 )
      goto LABEL_22;
    v39 = v13;
    if ( v13 <= 0x40 )
      goto LABEL_51;
    sub_C43780(&v38, &v34);
LABEL_30:
    v37 = v33;
    if ( v33 > 0x40 )
      sub_C43780(&v36, &v32);
    else
      v36 = v32;
    sub_AADC30((__int64)&v40, (__int64)&v36, &v38);
    v15 = (unsigned int)v45;
    v16 = v45;
    if ( (unsigned __int64)(unsigned int)v45 + 1 > HIDWORD(v45) )
    {
      if ( v44 > &v40 || &v40 >= &v44[4 * (unsigned int)v45] )
      {
        sub_9D5330((__int64)&v44, (unsigned int)v45 + 1LL);
        v15 = (unsigned int)v45;
        v17 = (unsigned __int64)v44;
        v18 = (char *)&v40;
        v16 = v45;
      }
      else
      {
        v27 = (unsigned __int64)v44;
        sub_9D5330((__int64)&v44, (unsigned int)v45 + 1LL);
        v17 = (unsigned __int64)v44;
        v15 = (unsigned int)v45;
        v16 = v45;
        v18 = (char *)&v40 + (_QWORD)v44 - v27;
      }
    }
    else
    {
      v17 = (unsigned __int64)v44;
      v18 = (char *)&v40;
    }
    v19 = v17 + 32 * v15;
    if ( v19 )
    {
      v20 = *((_DWORD *)v18 + 2);
      *((_DWORD *)v18 + 2) = 0;
      *(_DWORD *)(v19 + 8) = v20;
      *(_QWORD *)v19 = *(_QWORD *)v18;
      v21 = *((_DWORD *)v18 + 6);
      *((_DWORD *)v18 + 6) = 0;
      v16 = v45;
      *(_DWORD *)(v19 + 24) = v21;
      *(_QWORD *)(v19 + 16) = *((_QWORD *)v18 + 2);
    }
    LODWORD(v45) = v16 + 1;
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    v13 = v35;
LABEL_22:
    v4 = v10;
    if ( (int)sub_C4C880(v11, v10) >= 0 )
      ++v6;
    else
      ++v31;
    if ( v13 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 >= *(unsigned int *)(v29 + 8) )
      break;
    v5 = *(unsigned int *)(a3 + 8);
  }
  v22 = v45;
  *a1 = v26;
  a1[1] = 0x200000000LL;
  if ( !v22 )
    goto LABEL_53;
  v4 = (unsigned __int64 *)&v44;
  sub_ABF400((__int64)a1, (unsigned __int64 *)&v44);
  v23 = v44;
  v24 = &v44[4 * (unsigned int)v45];
  if ( v44 != v24 )
  {
    do
    {
      v24 -= 4;
      if ( *((_DWORD *)v24 + 6) > 0x40u )
      {
        v25 = v24[2];
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
      if ( *((_DWORD *)v24 + 2) > 0x40u && *v24 )
        j_j___libc_free_0_0(*v24);
    }
    while ( v23 != v24 );
LABEL_53:
    v23 = v44;
  }
  if ( v23 != (__int64 *)v46 )
    _libc_free(v23, v4);
  return a1;
}
