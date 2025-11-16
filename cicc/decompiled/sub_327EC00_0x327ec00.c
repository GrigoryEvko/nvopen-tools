// Function: sub_327EC00
// Address: 0x327ec00
//
__int64 __fastcall sub_327EC00(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int16 *v7; // rax
  unsigned __int16 v8; // r14
  _DWORD *v9; // r15
  int v10; // r9d
  __int64 v11; // rax
  _QWORD *i; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int128 v21; // rax
  __int128 v22; // kr00_16
  int v23; // r9d
  __int128 v24; // rax
  int v25; // r9d
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rax
  int v29; // eax
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rdi
  _QWORD *j; // rdi
  __int64 v35; // rdi
  _QWORD *k; // r15
  __int64 v37; // r13
  _BYTE *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rsi
  __int128 *v42; // r14
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-C0h]
  unsigned __int16 v45; // [rsp+14h] [rbp-ACh]
  bool v46; // [rsp+17h] [rbp-A9h]
  bool v47; // [rsp+17h] [rbp-A9h]
  __int64 v48; // [rsp+18h] [rbp-A8h]
  __int64 v49; // [rsp+18h] [rbp-A8h]
  char v50; // [rsp+18h] [rbp-A8h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  void *v53; // [rsp+28h] [rbp-98h]
  __int64 v54; // [rsp+30h] [rbp-90h] BYREF
  int v55; // [rsp+38h] [rbp-88h]
  __int64 v56; // [rsp+40h] [rbp-80h]
  void *v57; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v58; // [rsp+58h] [rbp-68h]
  __int64 v59; // [rsp+70h] [rbp-50h] BYREF
  int v60; // [rsp+78h] [rbp-48h]

  v3 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), 0);
  if ( !v3 )
    return 0;
  v4 = v3;
  v5 = *(_QWORD *)a1;
  v55 = *(_DWORD *)(a2 + 28);
  v6 = *(_QWORD *)(v5 + 1024);
  v54 = v5;
  v56 = v6;
  *(_QWORD *)(v5 + 1024) = &v54;
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *v7;
  v45 = *v7;
  v44 = *((_QWORD *)v7 + 1);
  v53 = sub_C33340();
  v9 = sub_C33320();
  if ( v8 != 12 )
  {
    if ( v45 != 13 )
      goto LABEL_10;
    v48 = *(_QWORD *)(v4 + 96);
    sub_C3B1B0((__int64)&v59, 0.3333333333333333);
    sub_C407B0(&v57, &v59, v9);
    sub_C338F0((__int64)&v59);
    sub_C41640((__int64 *)&v57, *(_DWORD **)(v48 + 24), 1, (bool *)&v59);
    v46 = 0;
    v11 = (__int64)v57;
    if ( *(void **)(v48 + 24) == v57 )
    {
      v32 = v48 + 24;
      if ( v53 == v57 )
        v46 = sub_C3E590(v32, (__int64)&v57);
      else
        v46 = sub_C33D00(v32, (__int64)&v57);
      v11 = (__int64)v57;
    }
    if ( v53 == (void *)v11 )
    {
      if ( v58 )
      {
        for ( i = &v58[3 * *(v58 - 1)]; v58 != i; i -= 3 )
          sub_91D830(i - 3);
        goto LABEL_9;
      }
      goto LABEL_34;
    }
LABEL_33:
    sub_C338F0((__int64)&v57);
    goto LABEL_34;
  }
  v51 = *(_QWORD *)(v4 + 96);
  sub_C3B1B0((__int64)&v59, 0.3333333432674408);
  sub_C407B0(&v57, &v59, v9);
  sub_C338F0((__int64)&v59);
  sub_C41640((__int64 *)&v57, *(_DWORD **)(v51 + 24), 1, (bool *)&v59);
  v46 = 0;
  v28 = (__int64)v57;
  if ( *(void **)(v51 + 24) == v57 )
  {
    v31 = v51 + 24;
    if ( v57 != v53 )
    {
      v46 = sub_C33D00(v31, (__int64)&v57);
      if ( v53 != v57 )
        goto LABEL_33;
      goto LABEL_42;
    }
    v46 = sub_C3E590(v31, (__int64)&v57);
    v28 = (__int64)v57;
  }
  if ( v53 != (void *)v28 )
    goto LABEL_33;
LABEL_42:
  if ( v58 )
  {
    for ( i = &v58[3 * *(v58 - 1)]; v58 != i; i -= 3 )
      sub_91D830(i - 3);
LABEL_9:
    j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    if ( !v46 )
      goto LABEL_10;
    goto LABEL_35;
  }
LABEL_34:
  if ( !v46 )
  {
LABEL_10:
    v49 = *(_QWORD *)(v4 + 96);
    sub_C3B1B0((__int64)&v59, 0.25);
    sub_C407B0(&v57, &v59, v9);
    sub_C338F0((__int64)&v59);
    sub_C41640((__int64 *)&v57, *(_DWORD **)(v49 + 24), 1, (bool *)&v59);
    v47 = 0;
    v13 = (__int64)v57;
    if ( *(void **)(v49 + 24) == v57 )
    {
      v33 = v49 + 24;
      if ( v53 != v57 )
      {
        v47 = sub_C33D00(v33, (__int64)&v57);
        if ( v53 != v57 )
          goto LABEL_12;
        goto LABEL_51;
      }
      v47 = sub_C3E590(v33, (__int64)&v57);
      v13 = (__int64)v57;
    }
    if ( v53 != (void *)v13 )
    {
LABEL_12:
      sub_C338F0((__int64)&v57);
      goto LABEL_13;
    }
LABEL_51:
    if ( v58 )
    {
      for ( j = &v58[3 * *(v58 - 1)]; v58 != j; j -= 3 )
        sub_91D830(j - 3);
      j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
    }
LABEL_13:
    v14 = *(_QWORD *)(v4 + 96);
    sub_C3B1B0((__int64)&v59, 0.75);
    sub_C407B0(&v57, &v59, v9);
    sub_C338F0((__int64)&v59);
    sub_C41640((__int64 *)&v57, *(_DWORD **)(v14 + 24), 1, (bool *)&v59);
    v16 = (__int64)v57;
    v50 = v47;
    if ( *(void **)(v14 + 24) == v57 )
    {
      v35 = v14 + 24;
      if ( v53 != v57 )
      {
        v50 = v47 | sub_C33D00(v35, (__int64)&v57);
        if ( v53 != v57 )
          goto LABEL_15;
        goto LABEL_57;
      }
      v50 = v47 | sub_C3E590(v35, (__int64)&v57);
      v16 = (__int64)v57;
    }
    if ( v53 != (void *)v16 )
    {
LABEL_15:
      sub_C338F0((__int64)&v57);
      goto LABEL_16;
    }
LABEL_57:
    if ( v58 )
    {
      for ( k = &v58[3 * *(v58 - 1)]; v58 != k; sub_91D830(k) )
        k -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
    }
LABEL_16:
    if ( v50 && ((*(_DWORD *)(a2 + 28) & 0x80u) != 0 || !v47) && (*(_DWORD *)(a2 + 28) & 0x440) == 0x440 )
    {
      v17 = *(_QWORD *)a1;
      v18 = 1;
      v19 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
      if ( v45 == 1 || v45 && (v18 = v45, *(_QWORD *)(v19 + 8LL * v45 + 112)) )
      {
        if ( (*(_BYTE *)(v19 + 500 * v18 + 6660) & 0xFB) == 0 && !a1[35] )
        {
          v20 = *(_QWORD *)(a2 + 80);
          v59 = v20;
          if ( v20 )
          {
            sub_B96E90((__int64)&v59, v20, 1);
            v17 = *(_QWORD *)a1;
          }
          v60 = *(_DWORD *)(a2 + 72);
          *(_QWORD *)&v21 = sub_33FAF80(v17, 246, (unsigned int)&v59, v45, v44, v15, *(_OWORD *)*(_QWORD *)(a2 + 40));
          v22 = v21;
          *(_QWORD *)&v24 = sub_33FAF80(*(_QWORD *)a1, 246, (unsigned int)&v59, v45, v44, v23, v21);
          if ( v47 )
            v26 = v24;
          else
            v26 = sub_3406EB0(*(_QWORD *)a1, 98, (unsigned int)&v59, v45, v44, v25, v22, v24);
          v27 = v59;
          if ( !v59 )
            goto LABEL_37;
          goto LABEL_30;
        }
      }
    }
LABEL_36:
    v26 = 0;
    goto LABEL_37;
  }
LABEL_35:
  v29 = *(_DWORD *)(a2 + 28);
  if ( (v29 & 0xC0) != 0xC0 )
    goto LABEL_36;
  if ( (v29 & 0x420) != 0x420 )
    goto LABEL_36;
  v37 = *(_QWORD *)a1;
  v38 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
  if ( (v38[32] & 2) != 0 )
    goto LABEL_36;
  if ( (*(_BYTE *)(*(_QWORD *)v38 + 48LL) & 0xC) == 0 )
    goto LABEL_36;
  if ( v45 )
  {
    v39 = *(_QWORD *)(v37 + 16);
    if ( *(_QWORD *)(v39 + 8LL * v45 + 112) )
    {
      v40 = 500LL * v45 + v39;
      if ( *(_BYTE *)(v40 + 6671) != 2 && *(_BYTE *)(v40 + 6661) == 2 )
        goto LABEL_36;
    }
  }
  v41 = *(_QWORD *)(a2 + 80);
  v42 = *(__int128 **)(a2 + 40);
  v59 = v41;
  if ( v41 )
    sub_B96E90((__int64)&v59, v41, 1);
  v60 = *(_DWORD *)(a2 + 72);
  v43 = sub_33FAF80(v37, 247, (unsigned int)&v59, v45, v44, v10, *v42);
  v27 = v59;
  v26 = v43;
  if ( !v59 )
    goto LABEL_37;
LABEL_30:
  sub_B91220((__int64)&v59, v27);
LABEL_37:
  *(_QWORD *)(v54 + 1024) = v56;
  return v26;
}
