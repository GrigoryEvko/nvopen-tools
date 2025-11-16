// Function: sub_1AD1F10
// Address: 0x1ad1f10
//
__int64 __fastcall sub_1AD1F10(__int64 a1, __int64 **a2)
{
  __int64 v2; // r8
  __int64 v4; // rax
  __int64 v5; // r13
  char *v6; // rax
  char *v7; // r12
  const void *v8; // r14
  int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rax
  size_t **v14; // r13
  __int64 *v15; // r12
  size_t *v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rdx
  char *v21; // r14
  unsigned __int64 v22; // rax
  size_t **v23; // r12
  size_t *v24; // r15
  size_t **v25; // rbx
  const void *v26; // rdi
  size_t *v27; // r14
  size_t v28; // rdx
  size_t v29; // rax
  int v30; // esi
  bool v31; // cc
  int v32; // eax
  size_t v33; // r9
  size_t v34; // rcx
  const void *v35; // rsi
  int v36; // eax
  __int64 v38; // rsi
  __int64 v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+0h] [rbp-60h]
  size_t v41; // [rsp+8h] [rbp-58h]
  size_t v42; // [rsp+8h] [rbp-58h]
  size_t v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  size_t v47; // [rsp+18h] [rbp-48h]
  size_t *v48; // [rsp+28h] [rbp-38h] BYREF

  v2 = a1;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v4 = *((unsigned int *)a2 + 3);
  if ( *((_DWORD *)a2 + 3) )
  {
    v5 = 8 * v4;
    v6 = (char *)sub_22077B0(8 * v4);
    v2 = a1;
    v7 = v6;
    v8 = *(const void **)a1;
    if ( (__int64)(*(_QWORD *)(v2 + 8) - *(_QWORD *)v2) > 0 )
    {
      memmove(v6, *(const void **)a1, *(_QWORD *)(a1 + 8) - *(_QWORD *)a1);
      v2 = a1;
      v38 = *(_QWORD *)(a1 + 16) - (_QWORD)v8;
    }
    else
    {
      if ( !v8 )
      {
LABEL_4:
        *(_QWORD *)v2 = v7;
        *(_QWORD *)(v2 + 8) = v7;
        *(_QWORD *)(v2 + 16) = &v7[v5];
        goto LABEL_5;
      }
      v38 = *(_QWORD *)(a1 + 16) - (_QWORD)v8;
    }
    v45 = v2;
    j_j___libc_free_0(v8, v38);
    v2 = v45;
    goto LABEL_4;
  }
LABEL_5:
  v9 = *((_DWORD *)a2 + 2);
  if ( v9 )
  {
    v10 = *a2;
    v11 = **a2;
    v12 = *a2;
    if ( !v11 || v11 == -8 )
    {
      do
      {
        do
        {
          v13 = v12[1];
          ++v12;
        }
        while ( v13 == -8 );
      }
      while ( !v13 );
    }
    v14 = *(size_t ***)(v2 + 8);
    v15 = &v10[v9];
    while ( v12 != v15 )
    {
      while ( 1 )
      {
        v16 = (size_t *)*v12;
        v48 = (size_t *)*v12;
        if ( v14 == *(size_t ***)(v2 + 16) )
        {
          v46 = v2;
          sub_1AD1D80(v2, v14, &v48);
          v2 = v46;
          v14 = *(size_t ***)(v46 + 8);
        }
        else
        {
          if ( v14 )
          {
            *v14 = v16;
            v14 = *(size_t ***)(v2 + 8);
          }
          *(_QWORD *)(v2 + 8) = ++v14;
        }
        v17 = v12[1];
        v18 = v12 + 1;
        if ( !v17 || v17 == -8 )
          break;
        ++v12;
        if ( v18 == v15 )
          goto LABEL_21;
      }
      v19 = v12 + 2;
      do
      {
        do
        {
          v20 = *v19;
          v12 = v19++;
        }
        while ( v20 == -8 );
      }
      while ( !v20 );
    }
  }
  else
  {
    v14 = *(size_t ***)(v2 + 8);
  }
LABEL_21:
  v21 = *(char **)v2;
  if ( *(size_t ***)v2 != v14 )
  {
    v44 = v2;
    _BitScanReverse64(&v22, ((char *)v14 - v21) >> 3);
    sub_1AD0A70(*(size_t ***)v2, v14, 2LL * (int)(63 - (v22 ^ 0x3F)));
    if ( (char *)v14 - v21 <= 128 )
    {
      sub_1AD0290(v21, (char *)v14);
      return v44;
    }
    v23 = (size_t **)(v21 + 128);
    sub_1AD0290(v21, v21 + 128);
    v2 = v44;
    if ( v21 + 128 != (char *)v14 )
    {
LABEL_24:
      v24 = *v23;
      v25 = v23;
      v26 = *v23 + 2;
      while ( 1 )
      {
        v27 = *(v25 - 1);
        v28 = v24[1];
        v29 = v27[1];
        v30 = *(_DWORD *)(v29 + 80);
        v31 = *(_DWORD *)(v28 + 80) <= v30;
        if ( *(_DWORD *)(v28 + 80) != v30
          || (v32 = *(_DWORD *)(v29 + 84), v31 = *(_DWORD *)(v28 + 84) <= v32, *(_DWORD *)(v28 + 84) != v32) )
        {
          if ( v31 )
            goto LABEL_34;
          goto LABEL_27;
        }
        v33 = *v27;
        v34 = *v24;
        v35 = v27 + 2;
        if ( *v27 < *v24 )
        {
          v47 = *v24;
          if ( !v33 )
            goto LABEL_34;
          v40 = v2;
          v42 = *v27;
          v36 = memcmp(v26, v35, v33);
          v33 = v42;
          v2 = v40;
          v34 = v47;
          if ( !v36 )
            goto LABEL_33;
        }
        else if ( !v34
               || (v39 = v2, v41 = *v27, v43 = *v24, v36 = memcmp(v26, v35, *v24), v34 = v43, v33 = v41, v2 = v39, !v36) )
        {
          if ( v33 == v34 )
            goto LABEL_34;
LABEL_33:
          if ( v33 <= v34 )
            goto LABEL_34;
          goto LABEL_27;
        }
        if ( v36 >= 0 )
        {
LABEL_34:
          ++v23;
          *v25 = v24;
          if ( v14 == v23 )
            return v2;
          goto LABEL_24;
        }
LABEL_27:
        *v25-- = v27;
      }
    }
  }
  return v2;
}
