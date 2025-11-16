// Function: sub_25AA1B0
// Address: 0x25aa1b0
//
void __fastcall sub_25AA1B0(size_t a1, __int64 a2, __int64 a3, size_t *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // r15
  __int64 v9; // rdx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  unsigned __int64 v13; // rdi
  _BYTE *v14; // rax
  int v15; // ebx
  _BYTE *v16; // rsi
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  char *v19; // r15
  size_t v20; // r13
  char *v21; // r12
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  _BYTE *v24; // rax
  int v25; // r12d
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  char *v28; // r15
  size_t v29; // r13
  char *v30; // rbx
  _QWORD *v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rdx
  unsigned __int64 v34; // [rsp+18h] [rbp-98h] BYREF
  char v35[8]; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v36; // [rsp+28h] [rbp-88h]
  char v37[8]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v38; // [rsp+48h] [rbp-68h]
  unsigned __int64 v39; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v40; // [rsp+68h] [rbp-48h]
  __int64 v41; // [rsp+70h] [rbp-40h]
  __int64 v42; // [rsp+78h] [rbp-38h]

  v6 = a3;
  if ( *(_BYTE *)a2 == 62 )
  {
    v32 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v32 != 3 )
      return;
    v11 = v32 & 0xFFFFFFFFFFFFFFF9LL | 4;
    v10 = *(_QWORD *)(a2 - 64) & 0xFFFFFFFFFFFFFFF9LL;
LABEL_37:
    v34 = v11;
    goto LABEL_10;
  }
  if ( (unsigned int)*(unsigned __int8 *)a2 - 29 > 0x21 )
  {
    if ( *(_BYTE *)a2 != 85 )
    {
      if ( *(_BYTE *)a2 != 86 )
        goto LABEL_20;
      v8 = *(_QWORD *)(a2 - 64);
      v9 = *(_QWORD *)(a2 - 32);
      v34 = a2 & 0xFFFFFFFFFFFFFFF9LL;
      v10 = v8 & 0xFFFFFFFFFFFFFFF9LL;
      v11 = v9 & 0xFFFFFFFFFFFFFFF9LL;
LABEL_10:
      sub_25A73A0((__int64)v37, a4, v11);
      sub_25A73A0((__int64)v35, a4, v10);
LABEL_11:
      sub_25A93A0(&v39, a1, (__int64)v35, (__int64)v37);
      v12 = sub_25A6840(v6, (__int64 *)&v34);
      v13 = v12[1];
      *(_DWORD *)v12 = v39;
      v12[1] = v40;
      v12[2] = v41;
      v12[3] = v42;
      v40 = 0;
      v41 = 0;
      v42 = 0;
      if ( v13 )
      {
        j_j___libc_free_0(v13);
        if ( v40 )
          j_j___libc_free_0(v40);
      }
      if ( v36 )
        j_j___libc_free_0(v36);
      if ( v38 )
        j_j___libc_free_0(v38);
      return;
    }
LABEL_42:
    sub_25A9D20(a1, a2, (__int64 *)a3, (__int64)a4, a5, a6);
    return;
  }
  switch ( *(_BYTE *)a2 )
  {
    case 0x22:
      goto LABEL_42;
    case 0x3D:
      a3 = *(_QWORD *)(a2 - 32);
      v34 = a2 & 0xFFFFFFFFFFFFFFF9LL;
      if ( *(_BYTE *)a3 == 3 )
      {
        sub_25A73A0((__int64)v37, a4, a3 & 0xFFFFFFFFFFFFFFF9LL | 4);
        sub_25A73A0((__int64)v35, a4, v34);
        goto LABEL_11;
      }
      v24 = *(_BYTE **)(a1 + 56);
      v16 = *(_BYTE **)(a1 + 48);
      v25 = *(_DWORD *)(a1 + 40);
      v26 = v24 - v16;
      if ( v24 == v16 )
      {
        v29 = 0;
        v28 = 0;
LABEL_31:
        v30 = &v28[v26];
        if ( v16 != v24 )
          memmove(v28, v16, v29);
        v31 = sub_25A6840(v6, (__int64 *)&v34);
        v23 = v31[1];
        *(_DWORD *)v31 = v25;
        v31[1] = v28;
        v31[2] = &v28[v29];
        v31[3] = v30;
        if ( !v23 )
          return;
LABEL_27:
        j_j___libc_free_0(v23);
        return;
      }
      if ( v26 <= 0x7FFFFFFFFFFFFFF8LL )
      {
        v27 = sub_22077B0(v26);
        v16 = *(_BYTE **)(a1 + 48);
        v28 = (char *)v27;
        v24 = *(_BYTE **)(a1 + 56);
        v29 = v24 - v16;
        goto LABEL_31;
      }
LABEL_45:
      sub_4261EA(a1, v16, a3);
    case 0x1E:
      v33 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
      if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v33 + 24) + 16LL) + 8LL) == 7 )
        return;
      v10 = 0;
      if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 0 )
        v10 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) & 0xFFFFFFFFFFFFFFF9LL;
      v11 = v33 & 0xFFFFFFFFFFFFFFF9LL | 2;
      goto LABEL_37;
  }
LABEL_20:
  if ( !*(_QWORD *)(a2 + 16) )
    return;
  v14 = *(_BYTE **)(a1 + 56);
  v15 = *(_DWORD *)(a1 + 40);
  v39 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  v16 = *(_BYTE **)(a1 + 48);
  v17 = v14 - v16;
  if ( v14 == v16 )
  {
    v20 = 0;
    v19 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_45;
    v18 = sub_22077B0(v17);
    v16 = *(_BYTE **)(a1 + 48);
    v19 = (char *)v18;
    v14 = *(_BYTE **)(a1 + 56);
    v20 = v14 - v16;
  }
  v21 = &v19[v17];
  if ( v16 != v14 )
    memmove(v19, v16, v20);
  v22 = sub_25A6840(v6, (__int64 *)&v39);
  v23 = v22[1];
  *(_DWORD *)v22 = v15;
  v22[1] = v19;
  v22[2] = &v19[v20];
  v22[3] = v21;
  if ( v23 )
    goto LABEL_27;
}
