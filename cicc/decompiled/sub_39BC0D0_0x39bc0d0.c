// Function: sub_39BC0D0
// Address: 0x39bc0d0
//
void __fastcall sub_39BC0D0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  char *v4; // rax
  char *v5; // rbx
  int v6; // edx
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 *v9; // rbx
  char *i; // rsi
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rax
  char *v15; // r8
  char *v16; // rdx
  int v17; // edi
  char *v18; // rax
  __int64 v19; // rsi
  _DWORD *v20; // rdx
  void *src; // [rsp+0h] [rbp-40h] BYREF
  char *v22; // [rsp+8h] [rbp-38h]
  char *v23; // [rsp+10h] [rbp-30h]

  v2 = *(unsigned int *)(a1 + 116);
  src = 0;
  v22 = 0;
  v23 = 0;
  if ( v2 )
  {
    v3 = 4 * v2;
    v4 = (char *)sub_22077B0(4 * v2);
    v5 = v4;
    if ( v22 - (_BYTE *)src > 0 )
    {
      memmove(v4, src, v22 - (_BYTE *)src);
      j_j___libc_free_0((unsigned __int64)src);
    }
    src = v5;
    v22 = v5;
    v23 = &v5[v3];
  }
  v6 = *(_DWORD *)(a1 + 112);
  if ( v6 )
  {
    v7 = *(__int64 **)(a1 + 104);
    v8 = *v7;
    v9 = v7;
    if ( *v7 != -8 )
      goto LABEL_8;
    do
    {
      do
      {
        v8 = v9[1];
        ++v9;
      }
      while ( v8 == -8 );
LABEL_8:
      ;
    }
    while ( !v8 );
    i = v22;
    v11 = &v7[v6];
    if ( v9 != v11 )
    {
      v12 = *v9;
      if ( v23 != v22 )
        goto LABEL_11;
LABEL_19:
      sub_B8BBF0((__int64)&src, i, (_DWORD *)(v12 + 16));
      for ( i = v22; ; v22 = i )
      {
        v13 = v9[1];
        v14 = v9 + 1;
        if ( v13 != -8 )
          goto LABEL_16;
        do
        {
          do
          {
            v13 = v14[1];
            ++v14;
          }
          while ( v13 == -8 );
LABEL_16:
          ;
        }
        while ( !v13 );
        if ( v14 == v11 )
          break;
        v9 = v14;
        v12 = *v14;
        if ( v23 == i )
          goto LABEL_19;
LABEL_11:
        if ( i )
        {
          *(_DWORD *)i = *(_DWORD *)(v12 + 16);
          i = v22;
        }
        i += 4;
      }
    }
  }
  else
  {
    i = v22;
  }
  v15 = (char *)src;
  if ( i - (_BYTE *)src > 4 )
  {
    qsort(src, (i - (_BYTE *)src) >> 2, 4u, (__compar_fn_t)sub_1DC3280);
    v15 = (char *)src;
    i = v22;
  }
  if ( v15 != i )
  {
    v16 = v15;
    while ( 1 )
    {
      v18 = v16;
      v16 += 4;
      if ( v16 == i )
        break;
      v17 = *((_DWORD *)v16 - 1);
      if ( v17 == *((_DWORD *)v18 + 1) )
      {
        if ( v18 != i )
        {
          if ( i == v18 + 8 )
          {
            i = v16;
          }
          else
          {
            v20 = v18 + 8;
            while ( 1 )
            {
              if ( *v20 != v17 )
              {
                *((_DWORD *)v18 + 1) = *v20;
                v18 += 4;
              }
              if ( i == (char *)++v20 )
                break;
              v17 = *(_DWORD *)v18;
            }
            v15 = (char *)src;
            i = v18 + 4;
          }
        }
        break;
      }
    }
  }
  v19 = (i - v15) >> 2;
  *(_DWORD *)(a1 + 148) = v19;
  if ( (unsigned int)v19 <= 0x400 )
  {
    if ( (unsigned int)v19 <= 0x10 )
    {
      if ( !(_DWORD)v19 )
        LODWORD(v19) = 1;
      *(_DWORD *)(a1 + 144) = v19;
    }
    else
    {
      *(_DWORD *)(a1 + 144) = (unsigned int)v19 >> 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 144) = (unsigned int)v19 >> 2;
  }
  if ( v15 )
    j_j___libc_free_0((unsigned __int64)v15);
}
