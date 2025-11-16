// Function: sub_3285B00
// Address: 0x3285b00
//
__int64 __fastcall sub_3285B00(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  int v11; // esi
  int v12; // eax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rbx
  int v16; // edx
  __int64 *v17; // r12
  __int64 v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rbx
  _QWORD *v21; // rax
  int v22; // ebx
  void *v23; // r13
  unsigned __int64 v24; // rdx
  size_t v25; // r12
  char v26; // di
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // [rsp+1Ch] [rbp-94h]
  char v32; // [rsp+26h] [rbp-8Ah]
  unsigned __int8 v33; // [rsp+27h] [rbp-89h]
  void *src; // [rsp+30h] [rbp-80h] BYREF
  __int64 v35; // [rsp+38h] [rbp-78h]
  _BYTE v36[112]; // [rsp+40h] [rbp-70h] BYREF

  v32 = a5;
  if ( *(_BYTE *)(a2 + 28) )
  {
    v8 = *(_QWORD **)(a2 + 8);
    v9 = &v8[*(unsigned int *)(a2 + 20)];
    if ( v8 == v9 )
      goto LABEL_9;
    while ( a1 != *v8 )
    {
      if ( v9 == ++v8 )
        goto LABEL_9;
    }
    return 1;
  }
  if ( sub_C8CA60(a2, a1) )
    return 1;
LABEL_9:
  src = v36;
  v35 = 0x800000000LL;
  v11 = *(_DWORD *)(a1 + 36);
  v12 = ~v11;
  if ( v11 >= -1 )
    v12 = *(_DWORD *)(a1 + 36);
  v30 = v12;
  LODWORD(v13) = *(_DWORD *)(a3 + 8);
  if ( !(_DWORD)v13 )
  {
    v33 = 0;
    v22 = 0;
    LODWORD(v13) = 0;
    goto LABEL_33;
  }
  while ( 2 )
  {
    v14 = (unsigned int)v13;
    v13 = (unsigned int)(v13 - 1);
    v15 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v14 - 8);
    *(_DWORD *)(a3 + 8) = v13;
    if ( v32 )
    {
      if ( *(_DWORD *)(v15 + 24) != 2 )
      {
        v16 = *(_DWORD *)(v15 + 36);
        LOBYTE(v14) = v16 > 0;
        if ( v30 > v16 && v16 > 0 && v30 > 0 )
        {
          v28 = (unsigned int)v35;
          v29 = (unsigned int)v35 + 1LL;
          if ( v29 > HIDWORD(v35) )
          {
            sub_C8D5F0((__int64)&src, v36, v29, 8u, a5, a6);
            v28 = (unsigned int)v35;
          }
          *((_QWORD *)src + v28) = v15;
          v13 = *(unsigned int *)(a3 + 8);
          LODWORD(v35) = v35 + 1;
          goto LABEL_28;
        }
      }
    }
    v17 = *(__int64 **)(v15 + 40);
    v18 = 5LL * *(unsigned int *)(v15 + 64);
    v19 = &v17[5 * *(unsigned int *)(v15 + 64)];
    if ( v17 == v19 )
      goto LABEL_26;
    v33 = 0;
    do
    {
      while ( 1 )
      {
        v20 = *v17;
        if ( !*(_BYTE *)(a2 + 28) )
          break;
        v21 = *(_QWORD **)(a2 + 8);
        v14 = *(unsigned int *)(a2 + 20);
        v18 = (__int64)&v21[v14];
        if ( v21 != (_QWORD *)v18 )
        {
          while ( v20 != *v21 )
          {
            if ( (_QWORD *)v18 == ++v21 )
              goto LABEL_45;
          }
          goto LABEL_23;
        }
LABEL_45:
        if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 16) )
          break;
        *(_DWORD *)(a2 + 20) = v14 + 1;
        *(_QWORD *)v18 = v20;
        ++*(_QWORD *)a2;
LABEL_40:
        v27 = *(unsigned int *)(a3 + 8);
        v14 = *(unsigned int *)(a3 + 12);
        if ( v27 + 1 > v14 )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v27 + 1, 8u, a5, a6);
          v27 = *(unsigned int *)(a3 + 8);
        }
        v18 = *(_QWORD *)a3;
        v17 += 5;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v27) = v20;
        ++*(_DWORD *)(a3 + 8);
        if ( a1 != v20 )
          goto LABEL_24;
LABEL_43:
        if ( v19 == v17 )
        {
          v33 = 1;
          v13 = *(unsigned int *)(a3 + 8);
          goto LABEL_30;
        }
        v33 = 1;
      }
      sub_C8CC70(a2, *v17, v18, v14, a5, a6);
      if ( (_BYTE)v18 )
        goto LABEL_40;
LABEL_23:
      v17 += 5;
      if ( a1 == v20 )
        goto LABEL_43;
LABEL_24:
      ;
    }
    while ( v19 != v17 );
    v13 = *(unsigned int *)(a3 + 8);
    if ( v33 )
      goto LABEL_30;
LABEL_26:
    if ( !a4 || a4 > *(_DWORD *)(a2 + 20) - *(_DWORD *)(a2 + 24) )
    {
LABEL_28:
      if ( !(_DWORD)v13 )
        break;
      continue;
    }
    break;
  }
  v33 = 0;
LABEL_30:
  v22 = v35;
  v23 = src;
  v24 = (unsigned int)v35 + v13;
  v25 = 8LL * (unsigned int)v35;
  if ( v24 > *(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v24, 8u, a5, a6);
    v13 = *(unsigned int *)(a3 + 8);
    if ( v25 )
    {
LABEL_32:
      memcpy((void *)(*(_QWORD *)a3 + 8 * v13), v23, v25);
      LODWORD(v13) = *(_DWORD *)(a3 + 8);
    }
  }
  else if ( v25 )
  {
    goto LABEL_32;
  }
LABEL_33:
  *(_DWORD *)(a3 + 8) = v22 + v13;
  if ( a4 )
  {
    v26 = v33;
    if ( a4 <= *(_DWORD *)(a2 + 20) - *(_DWORD *)(a2 + 24) )
      v26 = 1;
    v33 = v26;
  }
  if ( src != v36 )
    _libc_free((unsigned __int64)src);
  return v33;
}
