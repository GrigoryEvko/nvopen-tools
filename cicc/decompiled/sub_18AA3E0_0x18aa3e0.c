// Function: sub_18AA3E0
// Address: 0x18aa3e0
//
void __fastcall sub_18AA3E0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int8 a5)
{
  size_t v5; // r12
  int *v6; // r14
  unsigned int v8; // esi
  char *v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // ecx
  char *v12; // rax
  char *v13; // r9
  int v14; // ecx
  _QWORD *v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 i; // r12
  __int64 j; // r15
  int *v28; // r15
  __int64 v29; // rdi
  size_t v30; // rbx
  unsigned __int8 v31; // bl
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // r9
  unsigned int v35; // edx
  char *v36; // rdi
  __int64 v37; // r8
  size_t v38; // rax
  const char *v39; // r15
  int *v40; // rbx
  int v41; // eax
  int v42; // ecx
  char *v43; // r11
  int v44; // edx
  int v45; // r15d
  __int64 v46; // rcx
  int v47; // r8d
  char *v48; // rdi
  int v49; // ecx
  int v50; // ecx
  int v53; // [rsp+28h] [rbp-118h]
  __int64 v56; // [rsp+40h] [rbp-100h]
  __int64 v57; // [rsp+58h] [rbp-E8h] BYREF
  char *v58[2]; // [rsp+60h] [rbp-E0h] BYREF
  char *nptr[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v60[24]; // [rsp+80h] [rbp-C0h] BYREF

  if ( *(_QWORD *)(a1 + 16) <= a4 )
    return;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(int **)a1;
  sub_16C1840(nptr);
  sub_16C1A90((int *)nptr, v6, v5);
  sub_16C1AA0(nptr, v58);
  v8 = *(_DWORD *)(a2 + 24);
  v9 = v58[0];
  if ( !v8 )
  {
    ++*(_QWORD *)a2;
LABEL_71:
    v8 *= 2;
    goto LABEL_72;
  }
  v10 = *(_QWORD *)(a2 + 8);
  v11 = (v8 - 1) & (37 * LODWORD(v58[0]));
  v12 = (char *)(v10 + 8LL * v11);
  v13 = *(char **)v12;
  if ( v58[0] == *(char **)v12 )
    goto LABEL_5;
  v47 = 1;
  v48 = 0;
  while ( v13 != (char *)-1LL )
  {
    if ( !v48 && v13 == (char *)-2LL )
      v48 = v12;
    v11 = (v8 - 1) & (v47 + v11);
    v12 = (char *)(v10 + 8LL * v11);
    v13 = *(char **)v12;
    if ( v58[0] == *(char **)v12 )
      goto LABEL_5;
    ++v47;
  }
  v49 = *(_DWORD *)(a2 + 16);
  if ( v48 )
    v12 = v48;
  ++*(_QWORD *)a2;
  v50 = v49 + 1;
  if ( 4 * v50 >= 3 * v8 )
    goto LABEL_71;
  if ( v8 - *(_DWORD *)(a2 + 20) - v50 <= v8 >> 3 )
  {
LABEL_72:
    sub_142F750(a2, v8);
    sub_1880D60(a2, (__int64 *)v58, nptr);
    v12 = nptr[0];
    v9 = v58[0];
    v50 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v50;
  if ( *(_QWORD *)v12 != -1 )
    --*(_DWORD *)(a2 + 20);
  *(_QWORD *)v12 = v9;
LABEL_5:
  v56 = *(_QWORD *)(a1 + 56);
  if ( v56 != a1 + 40 )
  {
    while ( 1 )
    {
      v14 = *(_DWORD *)(v56 + 56);
      if ( v14 )
      {
        v15 = *(_QWORD **)(v56 + 48);
        if ( *v15 != -8 && *v15 )
        {
          v18 = *(__int64 **)(v56 + 48);
        }
        else
        {
          v16 = v15 + 1;
          do
          {
            do
            {
              v17 = *v16;
              v18 = v16++;
            }
            while ( !v17 );
          }
          while ( v17 == -8 );
        }
        v19 = &v15[v14];
        if ( v18 != v19 )
          break;
      }
LABEL_21:
      v56 = sub_220EF30(v56);
      if ( a1 + 40 == v56 )
        goto LABEL_22;
    }
    v20 = v18;
    while ( 1 )
    {
      v21 = *v20;
      if ( *(_QWORD *)(*v20 + 8) > a4 )
      {
        v28 = (int *)(v21 + 16);
        v29 = sub_16321A0(a3, v21 + 16, *(_QWORD *)v21);
        if ( !v29 || !sub_1626D20(v29) )
          break;
      }
LABEL_15:
      v22 = v20[1];
      v23 = v20 + 1;
      if ( v22 != -8 && v22 )
      {
        ++v20;
        if ( v23 == v19 )
          goto LABEL_21;
      }
      else
      {
        v24 = v20 + 2;
        do
        {
          do
          {
            v25 = *v24;
            v20 = v24++;
          }
          while ( !v25 );
        }
        while ( v25 == -8 );
        if ( v20 == v19 )
          goto LABEL_21;
      }
    }
    if ( a5 )
    {
      nptr[0] = (char *)v60;
      v38 = strlen((const char *)(v21 + 16));
      sub_18A3750((__int64 *)nptr, (_BYTE *)(v21 + 16), (__int64)v28 + v38);
      v39 = nptr[0];
      v40 = __errno_location();
      v41 = *v40;
      *v40 = 0;
      v53 = v41;
      v32 = strtol(v39, v58, 10);
      if ( v39 == v58[0] )
        sub_426290((__int64)"stol");
      if ( *v40 == 34 )
        sub_426320((__int64)"stol");
      if ( !*v40 )
        *v40 = v53;
      v31 = a5;
    }
    else
    {
      v30 = *(_QWORD *)v21;
      sub_16C1840(nptr);
      sub_16C1A90((int *)nptr, v28, v30);
      v31 = 0;
      sub_16C1AA0(nptr, v58);
      v32 = (__int64)v58[0];
    }
    v57 = v32;
    v33 = *(_DWORD *)(a2 + 24);
    if ( v33 )
    {
      v34 = *(_QWORD *)(a2 + 8);
      v35 = (v33 - 1) & (37 * v32);
      v36 = (char *)(v34 + 8LL * v35);
      v37 = *(_QWORD *)v36;
      if ( v32 == *(_QWORD *)v36 )
      {
LABEL_33:
        if ( v31 && (_QWORD *)nptr[0] != v60 )
          j_j___libc_free_0(nptr[0], v60[0] + 1LL);
        goto LABEL_15;
      }
      v42 = 1;
      v43 = 0;
      while ( v37 != -1 )
      {
        if ( !v43 && v37 == -2 )
          v43 = v36;
        v45 = v42 + 1;
        v46 = (v33 - 1) & (v35 + v42);
        v36 = (char *)(v34 + 8 * v46);
        v35 = v46;
        v37 = *(_QWORD *)v36;
        if ( v32 == *(_QWORD *)v36 )
          goto LABEL_33;
        v42 = v45;
      }
      if ( !v43 )
        v43 = v36;
      ++*(_QWORD *)a2;
      v44 = *(_DWORD *)(a2 + 16) + 1;
      if ( 4 * v44 < 3 * v33 )
      {
        if ( v33 - *(_DWORD *)(a2 + 20) - v44 <= v33 >> 3 )
        {
          sub_142F750(a2, v33);
          sub_1880D60(a2, &v57, v58);
          v43 = v58[0];
          v32 = v57;
          v44 = *(_DWORD *)(a2 + 16) + 1;
        }
        goto LABEL_50;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_142F750(a2, 2 * v33);
    sub_1880D60(a2, &v57, v58);
    v43 = v58[0];
    v32 = v57;
    v44 = *(_DWORD *)(a2 + 16) + 1;
LABEL_50:
    *(_DWORD *)(a2 + 16) = v44;
    if ( *(_QWORD *)v43 != -1 )
      --*(_DWORD *)(a2 + 20);
    *(_QWORD *)v43 = v32;
    goto LABEL_33;
  }
LABEL_22:
  for ( i = *(_QWORD *)(a1 + 104); a1 + 88 != i; i = sub_220EF30(i) )
  {
    for ( j = *(_QWORD *)(i + 64); i + 48 != j; j = sub_220EF30(j) )
      sub_18AA3E0(j + 64, a2, a3, a4, a5);
  }
}
