// Function: sub_AC04E0
// Address: 0xac04e0
//
void __fastcall sub_AC04E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  unsigned __int64 v4; // rbx
  __int64 v5; // r13
  _BYTE *v6; // r13
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rbx
  _BYTE *v11; // r12
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // r15
  unsigned int v18; // eax
  unsigned int v19; // eax
  unsigned __int64 v20; // rsi
  __int64 v21; // r15
  __int64 *v22; // r13
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // edx
  unsigned __int64 v29; // r13
  __int64 v30; // [rsp+10h] [rbp-120h]
  unsigned __int64 v32; // [rsp+38h] [rbp-F8h]
  _BYTE **v33; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v34; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v35; // [rsp+58h] [rbp-D8h]
  __int64 v36; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v37; // [rsp+68h] [rbp-C8h]
  __int64 v38; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-B8h]
  __int64 v40; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v41; // [rsp+88h] [rbp-A8h]
  __int64 v42; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+98h] [rbp-98h]
  __int64 v44; // [rsp+A0h] [rbp-90h]
  unsigned int v45; // [rsp+A8h] [rbp-88h]
  _BYTE *v46; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v47; // [rsp+B8h] [rbp-78h]
  _BYTE v48[112]; // [rsp+C0h] [rbp-70h] BYREF

  if ( sub_AAF7D0(a2) )
    return;
  v3 = *(unsigned int *)(a1 + 8);
  if ( !(_DWORD)v3 )
    return;
  v4 = *(_QWORD *)a1;
  v5 = 32 * v3;
  if ( (int)sub_C4C880(*(_QWORD *)a1 + v5 - 16, a2) <= 0 || (int)sub_C4C880(a2 + 16, v4) <= 0 )
    return;
  v33 = &v46;
  v46 = v48;
  v32 = v4 + v5;
  v47 = 0x200000000LL;
  do
  {
    v6 = (_BYTE *)v4;
    if ( (int)sub_C4C880(a2 + 16, v4) <= 0 || (int)sub_C4C880(v4 + 16, a2) <= 0 )
    {
      v13 = (unsigned int)v47;
      v14 = (__int64)v46;
      v15 = (unsigned int)v47 + 1LL;
      v16 = v47;
      if ( v15 > HIDWORD(v47) )
      {
        if ( (unsigned __int64)v46 > v4 || (unsigned __int64)&v46[32 * (unsigned int)v47] <= v4 )
        {
          sub_9D5330((__int64)&v46, v15);
          v13 = (unsigned int)v47;
          v14 = (__int64)v46;
          v16 = v47;
        }
        else
        {
          v29 = v4 - (_QWORD)v46;
          sub_9D5330((__int64)&v46, v15);
          v14 = (__int64)v46;
          v13 = (unsigned int)v47;
          v6 = &v46[v29];
          v16 = v47;
        }
      }
      v17 = v14 + 32 * v13;
      if ( v17 )
      {
        v18 = *((_DWORD *)v6 + 2);
        *(_DWORD *)(v17 + 8) = v18;
        if ( v18 > 0x40 )
          sub_C43780(v17, v6);
        else
          *(_QWORD *)v17 = *(_QWORD *)v6;
        v19 = *((_DWORD *)v6 + 6);
        *(_DWORD *)(v17 + 24) = v19;
        if ( v19 > 0x40 )
          sub_C43780(v17 + 16, v6 + 16);
        else
          *(_QWORD *)(v17 + 16) = *((_QWORD *)v6 + 2);
        v16 = v47;
      }
      LODWORD(v47) = v16 + 1;
      goto LABEL_18;
    }
    v7 = sub_C4C880(v4, a2);
    if ( v7 > 0 )
    {
      if ( (int)sub_C4C880(a2, v4) <= 0 )
      {
        if ( (int)sub_C4C880(v4 + 16, a2 + 16) <= 0 )
          goto LABEL_18;
        if ( (int)sub_C4C880(v4, a2 + 16) > 0 )
        {
LABEL_11:
          v37 = *(_DWORD *)(a2 + 8);
          if ( v37 <= 0x40 )
            goto LABEL_12;
          goto LABEL_43;
        }
LABEL_51:
        v37 = *(_DWORD *)(v4 + 24);
        if ( v37 > 0x40 )
          sub_C43780(&v36, v4 + 16);
        else
          v36 = *(_QWORD *)(v4 + 16);
        v35 = *(_DWORD *)(a2 + 24);
        if ( v35 <= 0x40 )
        {
          v8 = *(_QWORD *)(a2 + 16);
LABEL_14:
          v34 = v8;
          if ( (int)sub_C4C880(&v34, &v36) >= 0 )
            goto LABEL_15;
          goto LABEL_59;
        }
        v20 = a2 + 16;
LABEL_45:
        sub_C43780(&v34, v20);
        if ( (int)sub_C4C880(&v34, &v36) >= 0 )
        {
LABEL_46:
          if ( v35 > 0x40 && v34 )
            j_j___libc_free_0_0(v34);
LABEL_15:
          if ( v37 > 0x40 )
          {
            v9 = v36;
            if ( v36 )
              goto LABEL_17;
          }
          goto LABEL_18;
        }
LABEL_59:
        v21 = (__int64)v33;
        v41 = v37;
        if ( v37 > 0x40 )
          sub_C43780(&v40, &v36);
        else
          v40 = v36;
        v39 = v35;
        if ( v35 > 0x40 )
          sub_C43780(&v38, &v34);
        else
          v38 = v34;
        v22 = &v42;
        sub_AADC30((__int64)&v42, (__int64)&v38, &v40);
        v23 = *(unsigned int *)(v21 + 8);
        v24 = v23 + 1;
        v25 = *(_DWORD *)(v21 + 8);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
        {
          if ( *(_QWORD *)v21 > (unsigned __int64)&v42
            || (v30 = *(_QWORD *)v21, (unsigned __int64)&v42 >= *(_QWORD *)v21 + 32 * v23) )
          {
            sub_9D5330(v21, v24);
            v23 = *(unsigned int *)(v21 + 8);
            v26 = *(_QWORD *)v21;
            v25 = *(_DWORD *)(v21 + 8);
          }
          else
          {
            sub_9D5330(v21, v24);
            v26 = *(_QWORD *)v21;
            v23 = *(unsigned int *)(v21 + 8);
            v22 = (__int64 *)((char *)&v42 + *(_QWORD *)v21 - v30);
            v25 = *(_DWORD *)(v21 + 8);
          }
        }
        else
        {
          v26 = *(_QWORD *)v21;
        }
        v27 = v26 + 32 * v23;
        if ( v27 )
        {
          *(_DWORD *)(v27 + 8) = *((_DWORD *)v22 + 2);
          *(_QWORD *)v27 = *v22;
          v28 = *((_DWORD *)v22 + 6);
          *((_DWORD *)v22 + 2) = 0;
          *(_DWORD *)(v27 + 24) = v28;
          *(_QWORD *)(v27 + 16) = v22[2];
          *((_DWORD *)v22 + 6) = 0;
          v25 = *(_DWORD *)(v21 + 8);
        }
        *(_DWORD *)(v21 + 8) = v25 + 1;
        if ( v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        if ( v43 > 0x40 && v42 )
          j_j___libc_free_0_0(v42);
        if ( v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
        if ( v41 > 0x40 && v40 )
          j_j___libc_free_0_0(v40);
        goto LABEL_46;
      }
LABEL_10:
      if ( (int)sub_C4C880(v4, a2 + 16) > 0 )
        goto LABEL_11;
      goto LABEL_51;
    }
    if ( (int)sub_C4C880(a2 + 16, v4 + 16) > 0 )
    {
      if ( (int)sub_C4C880(a2, v4) <= 0 && (int)sub_C4C880(v4 + 16, a2 + 16) <= 0 )
        goto LABEL_18;
      if ( v7 )
      {
        v37 = *(_DWORD *)(a2 + 8);
        if ( v37 <= 0x40 )
        {
LABEL_12:
          v36 = *(_QWORD *)a2;
          v35 = *(_DWORD *)(v4 + 8);
          if ( v35 <= 0x40 )
          {
LABEL_13:
            v8 = *(_QWORD *)v4;
            goto LABEL_14;
          }
          goto LABEL_44;
        }
LABEL_43:
        sub_C43780(&v36, a2);
        v35 = *(_DWORD *)(v4 + 8);
        if ( v35 <= 0x40 )
          goto LABEL_13;
LABEL_44:
        v20 = v4;
        goto LABEL_45;
      }
      goto LABEL_10;
    }
    v43 = *(_DWORD *)(a2 + 8);
    if ( v43 > 0x40 )
      sub_C43780(&v42, a2);
    else
      v42 = *(_QWORD *)a2;
    v41 = *(_DWORD *)(v4 + 8);
    if ( v41 > 0x40 )
      sub_C43780(&v40, v4);
    else
      v40 = *(_QWORD *)v4;
    sub_AC00B0((__int64 *)&v33, &v40, (__int64)&v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    v43 = *(_DWORD *)(v4 + 24);
    if ( v43 > 0x40 )
      sub_C43780(&v42, v4 + 16);
    else
      v42 = *(_QWORD *)(v4 + 16);
    v41 = *(_DWORD *)(a2 + 24);
    if ( v41 > 0x40 )
      sub_C43780(&v40, a2 + 16);
    else
      v40 = *(_QWORD *)(a2 + 16);
    sub_AC00B0((__int64 *)&v33, &v40, (__int64)&v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v43 > 0x40 )
    {
      v9 = v42;
      if ( v42 )
LABEL_17:
        j_j___libc_free_0_0(v9);
    }
LABEL_18:
    v4 += 32LL;
  }
  while ( v32 != v4 );
  sub_ABF850(a1, (__int64 *)&v46);
  v10 = (__int64)v46;
  v11 = &v46[32 * (unsigned int)v47];
  if ( v46 != v11 )
  {
    do
    {
      v11 -= 32;
      if ( *((_DWORD *)v11 + 6) > 0x40u )
      {
        v12 = *((_QWORD *)v11 + 2);
        if ( v12 )
          j_j___libc_free_0_0(v12);
      }
      if ( *((_DWORD *)v11 + 2) > 0x40u && *(_QWORD *)v11 )
        j_j___libc_free_0_0(*(_QWORD *)v11);
    }
    while ( (_BYTE *)v10 != v11 );
    v11 = v46;
  }
  if ( v11 != v48 )
    _libc_free(v11, &v46);
}
