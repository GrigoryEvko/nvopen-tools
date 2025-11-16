// Function: sub_9974C0
// Address: 0x9974c0
//
char __fastcall sub_9974C0(__int64 a1, __int64 a2, unsigned __int8 *a3, char a4, int a5, __int64 *a6)
{
  unsigned int v8; // r12d
  __int64 v9; // r15
  int v10; // r14d
  __int64 v11; // rax
  unsigned int v12; // r12d
  unsigned int v13; // edx
  int v14; // eax
  bool v15; // al
  unsigned int v16; // ecx
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned int v19; // edi
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned int v23; // [rsp+Ch] [rbp-94h]
  unsigned int v24; // [rsp+Ch] [rbp-94h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  unsigned int v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+10h] [rbp-90h]
  unsigned int v30; // [rsp+18h] [rbp-88h]
  unsigned int v31; // [rsp+18h] [rbp-88h]
  unsigned int v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  unsigned int v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  __int64 v36; // [rsp+20h] [rbp-80h]
  unsigned int v37; // [rsp+20h] [rbp-80h]
  __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+38h] [rbp-68h]
  __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-58h]
  __int64 v43; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-48h]
  __int64 v45; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+68h] [rbp-38h]

  v8 = *(_DWORD *)(a1 + 8);
  if ( v8 <= 0x40 )
  {
    v9 = a1 + 16;
    v10 = sub_39FAC40(*(_QWORD *)a1);
    if ( *(_DWORD *)(a1 + 24) <= 0x40u )
      goto LABEL_3;
LABEL_6:
    LODWORD(v11) = sub_C44630(v9);
    if ( (_DWORD)v11 + v10 == v8 )
      return v11;
    goto LABEL_7;
  }
  v9 = a1 + 16;
  v10 = sub_C44630(a1);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
    goto LABEL_6;
LABEL_3:
  LODWORD(v11) = sub_39FAC40(*(_QWORD *)(a1 + 16));
  if ( (_DWORD)v11 + v10 == v8 )
    return v11;
LABEL_7:
  v44 = v8;
  if ( v8 > 0x40 )
  {
    sub_C43690(&v43, 0, 0);
    v46 = v8;
    sub_C43690(&v45, 0, 0);
  }
  else
  {
    v43 = 0;
    v46 = v8;
    v45 = 0;
  }
  v12 = a5 + 1;
  sub_996C70((__int64)a3, a2, &v43, a5 + 1, a6, a4);
  v13 = v44;
  if ( v44 <= 0x40 )
  {
    v15 = v43 == 0;
  }
  else
  {
    v35 = v44;
    v14 = sub_C444A0(&v43);
    v13 = v35;
    v15 = v35 == v14;
  }
  v16 = v46;
  if ( v15 )
  {
    if ( v46 <= 0x40 )
    {
      v11 = v45;
      if ( !v45 )
        goto LABEL_18;
      goto LABEL_22;
    }
    v30 = v46;
    LODWORD(v11) = sub_C444A0(&v45);
    if ( (_DWORD)v11 == v30 )
      goto LABEL_16;
    v40 = v30;
  }
  else
  {
    v40 = v46;
    if ( v46 <= 0x40 )
    {
      v11 = v45;
LABEL_22:
      v17 = *(_QWORD *)(a1 + 16) | v11;
      v39 = v17;
      goto LABEL_23;
    }
  }
  sub_C43780(&v39, &v45);
  v16 = v40;
  if ( v40 <= 0x40 )
  {
    v11 = v39;
    v13 = v44;
    goto LABEL_22;
  }
  sub_C43BD0(&v39, v9);
  v16 = v40;
  v17 = v39;
  v13 = v44;
LABEL_23:
  v40 = 0;
  v42 = v13;
  if ( v13 > 0x40 )
  {
    v27 = v16;
    v33 = v17;
    sub_C43780(&v41, &v43);
    v13 = v42;
    v17 = v33;
    v16 = v27;
    if ( v42 <= 0x40 )
    {
      v19 = v40;
      v18 = v41 | *(_QWORD *)a1;
    }
    else
    {
      sub_C43BD0(&v41, a1);
      v13 = v42;
      v18 = v41;
      v19 = v40;
      v16 = v27;
      v17 = v33;
    }
    if ( v19 > 0x40 && v39 )
    {
      v24 = v16;
      v28 = v17;
      v34 = v13;
      j_j___libc_free_0_0(v39);
      v16 = v24;
      v17 = v28;
      v13 = v34;
    }
  }
  else
  {
    v18 = *(_QWORD *)a1 | v43;
  }
  if ( v44 > 0x40 && v43 )
  {
    v23 = v16;
    v26 = v17;
    v31 = v13;
    j_j___libc_free_0_0(v43);
    v16 = v23;
    v17 = v26;
    v13 = v31;
  }
  v43 = v18;
  v44 = v13;
  if ( v46 > 0x40 && v45 )
  {
    v32 = v16;
    v36 = v17;
    j_j___libc_free_0_0(v45);
    v13 = v44;
    v16 = v32;
    v17 = v36;
  }
  v45 = v17;
  v46 = v16;
  if ( v13 <= 0x40 )
  {
    v11 = v43 & v17;
    if ( v11 )
      goto LABEL_35;
  }
  else
  {
    v37 = v16;
    LOBYTE(v11) = sub_C446A0(&v43, &v45);
    v16 = v37;
    if ( (_BYTE)v11 )
      goto LABEL_35;
  }
  LOBYTE(v11) = sub_98EF80(a3, a6[4], a6[5], a6[3], v12);
  if ( (_BYTE)v11 )
  {
    if ( *(_DWORD *)(a1 + 8) <= 0x40u && v44 <= 0x40 )
    {
      v21 = v43;
      *(_DWORD *)(a1 + 8) = v44;
      *(_QWORD *)a1 = v21;
    }
    else
    {
      sub_C43990(a1, &v43);
    }
    if ( *(_DWORD *)(a1 + 24) <= 0x40u && (LOBYTE(v11) = v46, v46 <= 0x40) )
    {
      v20 = v45;
      *(_DWORD *)(a1 + 24) = v46;
      *(_QWORD *)(a1 + 16) = v20;
    }
    else
    {
      LOBYTE(v11) = sub_C43990(v9, &v45);
      if ( v46 > 0x40 )
        goto LABEL_16;
    }
    goto LABEL_18;
  }
  v16 = v46;
LABEL_35:
  if ( v16 > 0x40 )
  {
LABEL_16:
    if ( v45 )
      LOBYTE(v11) = j_j___libc_free_0_0(v45);
  }
LABEL_18:
  if ( v44 > 0x40 && v43 )
    LOBYTE(v11) = j_j___libc_free_0_0(v43);
  return v11;
}
