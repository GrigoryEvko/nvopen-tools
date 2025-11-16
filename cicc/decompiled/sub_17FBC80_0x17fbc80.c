// Function: sub_17FBC80
// Address: 0x17fbc80
//
void __fastcall sub_17FBC80(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r14
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // rcx
  char v17; // al
  __int64 *v18; // rsi
  __int64 *v19; // rcx
  unsigned __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 *v22; // rdx
  const void *v23; // [rsp+0h] [rbp-C0h]
  __int64 *v25; // [rsp+10h] [rbp-B0h]
  __int64 v27; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v28; // [rsp+28h] [rbp-98h]
  __int64 *v29; // [rsp+30h] [rbp-90h]
  __int64 v30; // [rsp+38h] [rbp-88h]
  int v31; // [rsp+40h] [rbp-80h]
  _BYTE v32[120]; // [rsp+48h] [rbp-78h] BYREF

  v28 = (__int64 *)v32;
  v3 = *a1;
  v29 = (__int64 *)v32;
  v4 = v3 + 8LL * *((unsigned int *)a1 + 2);
  v27 = 0;
  v30 = 8;
  v31 = 0;
  if ( v3 == v4 )
  {
    *((_DWORD *)a1 + 2) = 0;
    return;
  }
  v23 = (const void *)(a2 + 16);
  do
  {
    v12 = *(_QWORD *)(v4 - 8);
    v13 = *(_QWORD *)(v12 - 24);
    if ( *(_BYTE *)(v12 + 16) != 55 )
    {
      v14 = sub_15F2050(v12);
      if ( !sub_17FBA90(v14, v13) )
        goto LABEL_9;
      v15 = v28;
      if ( v29 == v28 )
      {
        v16 = &v28[HIDWORD(v30)];
        if ( v28 == v16 )
        {
          v22 = v28;
        }
        else
        {
          do
          {
            if ( v13 == *v15 )
              break;
            ++v15;
          }
          while ( v16 != v15 );
          v22 = &v28[HIDWORD(v30)];
        }
      }
      else
      {
        v25 = &v29[(unsigned int)v30];
        v15 = sub_16CC9F0((__int64)&v27, v13);
        v16 = v25;
        if ( v13 == *v15 )
        {
          if ( v29 == v28 )
            v22 = &v29[HIDWORD(v30)];
          else
            v22 = &v29[(unsigned int)v30];
        }
        else
        {
          if ( v29 != v28 )
          {
            v15 = &v29[(unsigned int)v30];
            goto LABEL_16;
          }
          v15 = &v29[HIDWORD(v30)];
          v22 = v15;
        }
      }
      for ( ; v22 != v15; ++v15 )
      {
        if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
LABEL_16:
      if ( v16 != v15 )
        goto LABEL_9;
      v17 = *(_BYTE *)(v13 + 16);
      if ( v17 == 56 )
      {
        v13 = *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
        v17 = *(_BYTE *)(v13 + 16);
      }
      if ( v17 == 3 )
      {
        if ( (*(_BYTE *)(v13 + 80) & 1) != 0 )
          goto LABEL_9;
      }
      else if ( v17 == 54 && (*(_QWORD *)(v13 + 48) || *(__int16 *)(v13 + 18) < 0) )
      {
        v21 = sub_1625790(v13, 1);
        if ( v21 )
        {
          if ( sub_14A7290(v21) )
            goto LABEL_9;
        }
      }
      goto LABEL_6;
    }
    v6 = sub_15F2050(v12);
    if ( !sub_17FBA90(v6, v13) )
      goto LABEL_9;
    v7 = v28;
    if ( v29 != v28 )
      goto LABEL_5;
    v18 = &v28[HIDWORD(v30)];
    if ( v28 != v18 )
    {
      v19 = 0;
      while ( *v7 != v13 )
      {
        if ( *v7 == -2 )
          v19 = v7;
        if ( v18 == ++v7 )
        {
          if ( !v19 )
            goto LABEL_53;
          *v19 = v13;
          --v31;
          ++v27;
          goto LABEL_6;
        }
      }
      goto LABEL_6;
    }
LABEL_53:
    if ( HIDWORD(v30) < (unsigned int)v30 )
    {
      ++HIDWORD(v30);
      *v18 = v13;
      ++v27;
    }
    else
    {
LABEL_5:
      sub_16CCBA0((__int64)&v27, v13);
    }
LABEL_6:
    v8 = *(_QWORD *)(v12 - 24);
    if ( *(_BYTE *)(sub_14AD280(v8, a3, 6u) + 16) != 53 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 12) )
        goto LABEL_51;
      goto LABEL_8;
    }
    if ( (unsigned __int8)sub_139D0F0(v8, 1) )
    {
      v11 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 12) )
      {
LABEL_51:
        sub_16CD150(a2, v23, 0, 8, v9, v10);
        v11 = *(unsigned int *)(a2 + 8);
      }
LABEL_8:
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v12;
      ++*(_DWORD *)(a2 + 8);
    }
LABEL_9:
    v4 -= 8;
  }
  while ( v3 != v4 );
  v20 = (unsigned __int64)v29;
  *((_DWORD *)a1 + 2) = 0;
  if ( v28 != (__int64 *)v20 )
    _libc_free(v20);
}
