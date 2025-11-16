// Function: sub_25A9D20
// Address: 0x25a9d20
//
void __fastcall sub_25A9D20(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // r14
  __int64 v8; // r15
  size_t *v9; // r13
  bool v10; // zf
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _BYTE *v14; // rax
  _BYTE *v15; // rsi
  int v16; // r12d
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  char *v19; // r15
  size_t v20; // r13
  char *v21; // rbx
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rbx
  int v28; // edx
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rax
  unsigned __int64 v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // [rsp+20h] [rbp-B0h]
  __int64 v37; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v38; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v39; // [rsp+38h] [rbp-98h] BYREF
  char v40[8]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v41; // [rsp+48h] [rbp-88h]
  char v42[8]; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v43; // [rsp+68h] [rbp-68h]
  unsigned __int64 v44; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int64 v45; // [rsp+88h] [rbp-48h]
  __int64 v46; // [rsp+90h] [rbp-40h]
  __int64 v47; // [rsp+98h] [rbp-38h]

  v6 = a1;
  v8 = *(_QWORD *)(a2 - 32);
  v37 = (__int64)a3;
  if ( !v8 || *(_BYTE *)v8 )
  {
    v38 = a2 & 0xFFFFFFFFFFFFFFF9LL;
LABEL_36:
    if ( *(_BYTE *)(a1 + 132) )
    {
      v33 = *(__int64 **)(a1 + 112);
      a4 = *(unsigned int *)(a1 + 124);
      a3 = &v33[a4];
      if ( v33 != a3 )
      {
        while ( a2 != *v33 )
        {
          if ( a3 == ++v33 )
            goto LABEL_45;
        }
        goto LABEL_5;
      }
LABEL_45:
      if ( (unsigned int)a4 < *(_DWORD *)(a1 + 120) )
      {
        *(_DWORD *)(a1 + 124) = a4 + 1;
        *a3 = a2;
        ++*(_QWORD *)(a1 + 104);
        goto LABEL_5;
      }
    }
    a1 += 104;
    sub_C8CC70(a1, a2, (__int64)a3, a4, a5, a6);
    goto LABEL_5;
  }
  v9 = (size_t *)a4;
  a4 = *(_QWORD *)(v8 + 24);
  v10 = *(_QWORD *)(a2 + 80) == a4;
  v38 = a2 & 0xFFFFFFFFFFFFFFF9LL;
  if ( !v10 )
    goto LABEL_36;
  a1 = v8;
  if ( (unsigned __int8)sub_310F860(v8) )
  {
    v24 = *(_QWORD *)(v8 + 80);
    if ( v24 )
      v24 -= 24;
    sub_25A5A10((__int64)v9, v24, a3, v11, v12, v13);
    if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v8, v24, v25, v26);
      v27 = *(_QWORD *)(v8 + 96);
      v36 = v27 + 40LL * *(_QWORD *)(v8 + 104);
      if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v8, v24, v34, v35);
        v27 = *(_QWORD *)(v8 + 96);
      }
    }
    else
    {
      v27 = *(_QWORD *)(v8 + 96);
      v36 = v27 + 40LL * *(_QWORD *)(v8 + 104);
    }
    for ( ; v36 != v27; v27 += 40 )
    {
      v28 = *(_DWORD *)(a2 + 4);
      v39 = v27 & 0xFFFFFFFFFFFFFFF9LL;
      sub_25A73A0(
        (__int64)v42,
        v9,
        *(_QWORD *)(a2 + 32 * (*(unsigned int *)(v27 + 32) - (unsigned __int64)(v28 & 0x7FFFFFF)))
      & 0xFFFFFFFFFFFFFFF9LL);
      sub_25A73A0((__int64)v40, v9, v39);
      sub_25A93A0(&v44, v6, (__int64)v40, (__int64)v42);
      v29 = sub_25A6840(v37, (__int64 *)&v39);
      v30 = v29[1];
      *(_DWORD *)v29 = v44;
      v29[1] = v45;
      v29[2] = v46;
      v29[3] = v47;
      v45 = 0;
      v46 = 0;
      v47 = 0;
      if ( v30 )
      {
        j_j___libc_free_0(v30);
        if ( v45 )
          j_j___libc_free_0(v45);
      }
      if ( v41 )
        j_j___libc_free_0(v41);
      if ( v43 )
        j_j___libc_free_0(v43);
    }
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 7 )
    {
      sub_25A73A0((__int64)v42, v9, v8 & 0xFFFFFFFFFFFFFFF9LL | 2);
      sub_25A73A0((__int64)v40, v9, v38);
      sub_25A93A0(&v44, v6, (__int64)v40, (__int64)v42);
      v31 = sub_25A6840(v37, (__int64 *)&v38);
      v32 = v31[1];
      *(_DWORD *)v31 = v44;
      v31[1] = v45;
      v31[2] = v46;
      v31[3] = v47;
      v45 = 0;
      v46 = 0;
      v47 = 0;
      if ( v32 )
      {
        j_j___libc_free_0(v32);
        if ( v45 )
          j_j___libc_free_0(v45);
      }
      if ( v41 )
        j_j___libc_free_0(v41);
      v23 = v43;
      if ( v43 )
        goto LABEL_13;
    }
    return;
  }
LABEL_5:
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) != 7 )
  {
    v14 = *(_BYTE **)(v6 + 56);
    v15 = *(_BYTE **)(v6 + 48);
    v16 = *(_DWORD *)(v6 + 40);
    v17 = v14 - v15;
    if ( v14 == v15 )
    {
      v20 = 0;
      v19 = 0;
    }
    else
    {
      if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(a1, v15, a3);
      v18 = sub_22077B0(v17);
      v15 = *(_BYTE **)(v6 + 48);
      v19 = (char *)v18;
      v14 = *(_BYTE **)(v6 + 56);
      v20 = v14 - v15;
    }
    v21 = &v19[v17];
    if ( v15 != v14 )
      memmove(v19, v15, v20);
    v22 = sub_25A6840(v37, (__int64 *)&v38);
    v23 = v22[1];
    *(_DWORD *)v22 = v16;
    v22[1] = v19;
    v22[2] = &v19[v20];
    v22[3] = v21;
    if ( v23 )
LABEL_13:
      j_j___libc_free_0(v23);
  }
}
