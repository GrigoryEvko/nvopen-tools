// Function: sub_2EB02C0
// Address: 0x2eb02c0
//
__int64 __fastcall sub_2EB02C0(__int64 *a1, __int64 a2, __int64 a3)
{
  void **v5; // rax
  void **v6; // rdx
  __int64 v7; // rbx
  void **v9; // rdx
  __int64 v10; // r8
  void **v11; // rsi
  __int64 **v12; // rcx
  __int64 **v13; // rax
  __int64 **v14; // rcx
  __int64 **v15; // rax
  __int64 **v16; // rax
  __int64 **v17; // rdx
  void **v18; // rdi
  void **v19; // rsi

  if ( *(_DWORD *)(a3 + 68) == *(_DWORD *)(a3 + 72) )
  {
    if ( *(_BYTE *)(a3 + 28) )
    {
      v16 = *(__int64 ***)(a3 + 8);
      v17 = &v16[*(unsigned int *)(a3 + 20)];
      if ( v16 != v17 )
      {
        while ( *v16 != &qword_4F82400 )
        {
          if ( v17 == ++v16 )
            goto LABEL_2;
        }
        return 0;
      }
    }
    else if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
    {
      return 0;
    }
  }
LABEL_2:
  if ( *(_BYTE *)(a3 + 76) )
  {
    v5 = *(void ***)(a3 + 56);
    v6 = &v5[*(unsigned int *)(a3 + 68)];
    if ( v5 != v6 )
    {
      while ( *v5 != &unk_50209C8 )
      {
        if ( v6 == ++v5 )
          goto LABEL_9;
      }
      goto LABEL_7;
    }
  }
  else if ( sub_C8CA60(a3 + 48, (__int64)&unk_50209C8) )
  {
    goto LABEL_7;
  }
LABEL_9:
  if ( *(_BYTE *)(a3 + 28) )
  {
    v9 = *(void ***)(a3 + 8);
    v10 = *(unsigned int *)(a3 + 20);
    v11 = &v9[v10];
    v12 = (__int64 **)v9;
    if ( v9 == v11 )
      goto LABEL_7;
    v13 = *(__int64 ***)(a3 + 8);
    while ( *v13 != &qword_4F82400 )
    {
      if ( v11 == (void **)++v13 )
        goto LABEL_33;
    }
    goto LABEL_14;
  }
  if ( !sub_C8CA60(a3, (__int64)&qword_4F82400) )
  {
    if ( *(_BYTE *)(a3 + 28) )
    {
      v9 = *(void ***)(a3 + 8);
      v10 = *(unsigned int *)(a3 + 20);
      v13 = (__int64 **)&v9[v10];
      v12 = (__int64 **)v9;
      if ( v13 == (__int64 **)v9 )
        goto LABEL_7;
LABEL_33:
      v18 = v9;
      while ( *v18 != &unk_50209C8 )
      {
        if ( v13 == (__int64 **)++v18 )
          goto LABEL_41;
      }
    }
    else
    {
      if ( sub_C8CA60(a3, (__int64)&unk_50209C8) )
        goto LABEL_29;
      if ( *(_BYTE *)(a3 + 28) )
      {
        v9 = *(void ***)(a3 + 8);
        v10 = *(unsigned int *)(a3 + 20);
        v13 = (__int64 **)&v9[v10];
        v12 = (__int64 **)v9;
        if ( v13 == (__int64 **)v9 )
          goto LABEL_7;
LABEL_41:
        v19 = v9;
        while ( *v12 != &qword_4F82400 )
        {
          if ( ++v12 == v13 )
            goto LABEL_61;
        }
      }
      else
      {
        if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
          goto LABEL_29;
        if ( !*(_BYTE *)(a3 + 28) )
        {
          if ( !sub_C8CA60(a3, (__int64)&unk_4F82428) )
            goto LABEL_7;
          goto LABEL_29;
        }
        v9 = *(void ***)(a3 + 8);
        v10 = *(unsigned int *)(a3 + 20);
        v13 = (__int64 **)&v9[v10];
        v19 = v9;
        if ( v13 == (__int64 **)v9 )
          goto LABEL_7;
LABEL_61:
        while ( *v19 != &unk_4F82428 )
        {
          if ( ++v19 == (void **)v13 )
            goto LABEL_7;
        }
      }
    }
LABEL_14:
    if ( *(_DWORD *)(a3 + 72) != *(_DWORD *)(a3 + 68) )
      goto LABEL_7;
    goto LABEL_15;
  }
LABEL_29:
  if ( *(_DWORD *)(a3 + 68) != *(_DWORD *)(a3 + 72) )
    goto LABEL_7;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v9 = *(void ***)(a3 + 8);
    v10 = *(unsigned int *)(a3 + 20);
LABEL_15:
    v14 = (__int64 **)&v9[v10];
    if ( v9 != (void **)v14 )
    {
      v15 = (__int64 **)v9;
      while ( *v15 != &qword_4F82400 )
      {
        if ( v14 == ++v15 )
          goto LABEL_52;
      }
      return 0;
    }
LABEL_7:
    v7 = *a1;
    sub_2398D90(v7 + 64);
    sub_2398F30(v7 + 32);
    return 1;
  }
  if ( sub_C8CA60(a3, (__int64)&qword_4F82400) )
    return 0;
  if ( *(_BYTE *)(a3 + 28) )
  {
    v9 = *(void ***)(a3 + 8);
    v15 = (__int64 **)&v9[*(unsigned int *)(a3 + 20)];
    if ( v9 == (void **)v15 )
      goto LABEL_7;
LABEL_52:
    while ( *v9 != &unk_4FDC268 )
    {
      if ( ++v9 == (void **)v15 )
        goto LABEL_7;
    }
    return 0;
  }
  else
  {
    if ( !sub_C8CA60(a3, (__int64)&unk_4FDC268) )
      goto LABEL_7;
    return 0;
  }
}
