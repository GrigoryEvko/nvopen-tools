// Function: sub_3102230
// Address: 0x3102230
//
void __fastcall sub_3102230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 *v11; // rax
  __int64 *v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  unsigned __int64 v18; // rcx
  _BYTE *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 *v25; // rax
  __int64 *v26; // rdx
  __int64 *v27; // rax
  __int64 v28; // rdx
  char v29; // dl
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  _BYTE *v32; // [rsp+10h] [rbp-60h] BYREF
  __int64 v33; // [rsp+18h] [rbp-58h]
  _BYTE v34[80]; // [rsp+20h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(a2 + 16);
  v32 = v34;
  v33 = 0x400000000LL;
  if ( !v8 )
  {
LABEL_23:
    v17 = v34;
    LODWORD(v18) = 0;
    goto LABEL_24;
  }
  while ( 1 )
  {
    v9 = *(_QWORD *)(v8 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v9 - 30) <= 0xAu )
      break;
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      goto LABEL_23;
  }
  v10 = *(_QWORD *)(v9 + 40);
  if ( !*(_BYTE *)(a1 + 84) )
    goto LABEL_18;
LABEL_4:
  v11 = *(__int64 **)(a1 + 64);
  v12 = &v11[*(unsigned int *)(a1 + 76)];
  if ( v11 != v12 )
  {
    while ( v10 != *v11 )
    {
      if ( v12 == ++v11 )
        goto LABEL_15;
    }
    if ( !*(_BYTE *)(a3 + 28) )
    {
LABEL_20:
      sub_C8CC70(a3, v10, (__int64)v12, a4, a5, a6);
      v14 = (unsigned int)v33;
      a4 = HIDWORD(v33);
      v15 = (unsigned int)v33 + 1LL;
      if ( v15 <= HIDWORD(v33) )
        goto LABEL_14;
      goto LABEL_21;
    }
LABEL_9:
    v13 = *(__int64 **)(a3 + 8);
    a4 = *(unsigned int *)(a3 + 20);
    v12 = &v13[a4];
    if ( v13 == v12 )
    {
LABEL_54:
      if ( (unsigned int)a4 >= *(_DWORD *)(a3 + 16) )
        goto LABEL_20;
      *(_DWORD *)(a3 + 20) = a4 + 1;
      *v12 = v10;
      ++*(_QWORD *)a3;
    }
    else
    {
      while ( v10 != *v13 )
      {
        if ( v12 == ++v13 )
          goto LABEL_54;
      }
    }
    v14 = (unsigned int)v33;
    a4 = HIDWORD(v33);
    v15 = (unsigned int)v33 + 1LL;
    if ( v15 <= HIDWORD(v33) )
      goto LABEL_14;
LABEL_21:
    sub_C8D5F0((__int64)&v32, v34, v15, 8u, a5, a6);
    v14 = (unsigned int)v33;
LABEL_14:
    *(_QWORD *)&v32[8 * v14] = v10;
    LODWORD(v33) = v33 + 1;
  }
LABEL_15:
  while ( 1 )
  {
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      break;
    v16 = *(_QWORD *)(v8 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v16 - 30) <= 0xAu )
    {
      v10 = *(_QWORD *)(v16 + 40);
      if ( *(_BYTE *)(a1 + 84) )
        goto LABEL_4;
LABEL_18:
      if ( sub_C8CA60(a1 + 56, v10) )
      {
        if ( !*(_BYTE *)(a3 + 28) )
          goto LABEL_20;
        goto LABEL_9;
      }
    }
  }
  v17 = v32;
  LODWORD(v18) = v33;
LABEL_24:
  v19 = &v17[8 * (unsigned int)v18];
  while ( (_DWORD)v18 )
  {
    v20 = *(_QWORD **)(a1 + 32);
    v18 = (unsigned int)(v18 - 1);
    v21 = *((_QWORD *)v19 - 1);
    v19 -= 8;
    LODWORD(v33) = v18;
    if ( v21 != *v20 )
    {
      v22 = *(_QWORD *)(v21 + 16);
      if ( !v22 )
        goto LABEL_24;
      while ( 1 )
      {
        v23 = *(_QWORD *)(v22 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
          break;
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
          goto LABEL_24;
      }
      v24 = *(_QWORD *)(v23 + 40);
      if ( !*(_BYTE *)(a1 + 84) )
        goto LABEL_42;
LABEL_30:
      v25 = *(__int64 **)(a1 + 64);
      v26 = &v25[*(unsigned int *)(a1 + 76)];
      if ( v25 == v26 )
        goto LABEL_39;
      while ( v24 != *v25 )
      {
        if ( v26 == ++v25 )
          goto LABEL_39;
      }
LABEL_34:
      if ( !*(_BYTE *)(a3 + 28) )
        goto LABEL_47;
      v27 = *(__int64 **)(a3 + 8);
      v18 = *(unsigned int *)(a3 + 20);
      v26 = &v27[v18];
      if ( v27 != v26 )
      {
        while ( v24 != *v27 )
        {
          if ( v26 == ++v27 )
            goto LABEL_51;
        }
        goto LABEL_39;
      }
LABEL_51:
      if ( (unsigned int)v18 < *(_DWORD *)(a3 + 16) )
      {
        *(_DWORD *)(a3 + 20) = v18 + 1;
        *v26 = v24;
        v30 = (unsigned int)v33;
        v18 = HIDWORD(v33);
        ++*(_QWORD *)a3;
        v31 = v30 + 1;
        if ( v30 + 1 > v18 )
        {
LABEL_53:
          sub_C8D5F0((__int64)&v32, v34, v31, 8u, a5, a6);
          v30 = (unsigned int)v33;
        }
LABEL_49:
        *(_QWORD *)&v32[8 * v30] = v24;
        LODWORD(v33) = v33 + 1;
      }
      else
      {
LABEL_47:
        sub_C8CC70(a3, v24, (__int64)v26, v18, a5, a6);
        if ( v29 )
        {
          v30 = (unsigned int)v33;
          v18 = HIDWORD(v33);
          v31 = (unsigned int)v33 + 1LL;
          if ( v31 > HIDWORD(v33) )
            goto LABEL_53;
          goto LABEL_49;
        }
      }
      while ( 1 )
      {
LABEL_39:
        v22 = *(_QWORD *)(v22 + 8);
        if ( !v22 )
        {
LABEL_44:
          v17 = v32;
          LODWORD(v18) = v33;
          goto LABEL_24;
        }
        while ( 1 )
        {
          v28 = *(_QWORD *)(v22 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v28 - 30) > 0xAu )
            break;
          v24 = *(_QWORD *)(v28 + 40);
          if ( *(_BYTE *)(a1 + 84) )
            goto LABEL_30;
LABEL_42:
          if ( sub_C8CA60(a1 + 56, v24) )
            goto LABEL_34;
          v22 = *(_QWORD *)(v22 + 8);
          if ( !v22 )
            goto LABEL_44;
        }
      }
    }
  }
  if ( v17 != v34 )
    _libc_free((unsigned __int64)v17);
}
