// Function: sub_16C3CF0
// Address: 0x16c3cf0
//
void __fastcall sub_16C3CF0(__int64 a1, __int64 a2, int a3)
{
  bool v4; // zf
  char *v5; // r13
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r9
  char v8; // al
  const char *v9; // rcx
  size_t v10; // r15
  const char *v11; // r8
  __int64 v12; // rax
  int v13; // edx
  void *v14; // rdi
  const char *v15; // rdi
  size_t v16; // rdx
  unsigned __int64 v17; // rax
  size_t v18; // rax
  int v19; // [rsp+10h] [rbp-70h]
  int src; // [rsp+18h] [rbp-68h]
  const char *srca; // [rsp+18h] [rbp-68h]
  const char *srcc; // [rsp+18h] [rbp-68h]
  const char *srcd; // [rsp+18h] [rbp-68h]
  const char *srcb; // [rsp+18h] [rbp-68h]
  const char *v25; // [rsp+20h] [rbp-60h] BYREF
  size_t n; // [rsp+28h] [rbp-58h]
  _BYTE v27[80]; // [rsp+30h] [rbp-50h] BYREF

  v4 = *(_BYTE *)(a2 + 17) == 1;
  v5 = *(char **)a1;
  v25 = v27;
  v6 = *(unsigned int *)(a1 + 8);
  n = 0x2000000000LL;
  v7 = v6;
  if ( v4 )
  {
    v8 = *(_BYTE *)(a2 + 16);
    if ( v8 == 1 )
    {
      v10 = 0;
      v11 = 0;
    }
    else
    {
      v9 = *(const char **)a2;
      switch ( v8 )
      {
        case 3:
          v10 = 0;
          if ( v9 )
          {
            v19 = a3;
            srcd = *(const char **)a2;
            v18 = strlen(*(const char **)a2);
            v7 = v6;
            a3 = v19;
            v9 = srcd;
            v10 = v18;
          }
          v11 = v9;
          break;
        case 4:
        case 5:
          v11 = *(const char **)v9;
          v10 = *((_QWORD *)v9 + 1);
          break;
        case 6:
          v10 = *((unsigned int *)v9 + 2);
          v11 = *(const char **)v9;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    src = a3;
    sub_16E2F40(a2, &v25);
    v10 = (unsigned int)n;
    v11 = v25;
    v7 = v6;
    a3 = src;
  }
  do
  {
    if ( !v6 )
      goto LABEL_7;
    --v6;
  }
  while ( v5[v6] != 46 );
  srcc = v11;
  v17 = sub_16C3BE0(v5, v7, a3);
  v11 = srcc;
  if ( v17 > v6 )
  {
LABEL_7:
    v12 = *(unsigned int *)(a1 + 8);
    goto LABEL_8;
  }
  *(_DWORD *)(a1 + 8) = v6;
  v12 = (unsigned int)v6;
LABEL_8:
  v13 = v12;
  if ( v10 )
  {
    if ( *v11 == 46 )
    {
      if ( v10 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v12 )
      {
LABEL_11:
        srca = v11;
        sub_16CD150(a1, a1 + 16, v12 + v10, 1);
        v11 = srca;
        v14 = (void *)(*(_QWORD *)a1 + *(unsigned int *)(a1 + 8));
LABEL_12:
        memcpy(v14, v11, v10);
        v13 = *(_DWORD *)(a1 + 8);
        goto LABEL_13;
      }
    }
    else
    {
      if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v12 )
      {
        srcb = v11;
        sub_16CD150(a1, a1 + 16, 0, 1);
        v12 = *(unsigned int *)(a1 + 8);
        v11 = srcb;
      }
      *(_BYTE *)(*(_QWORD *)a1 + v12) = 46;
      v12 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      v16 = *(unsigned int *)(a1 + 12) - v12;
      *(_DWORD *)(a1 + 8) = v12;
      if ( v10 > v16 )
        goto LABEL_11;
    }
    v14 = (void *)(*(_QWORD *)a1 + v12);
    goto LABEL_12;
  }
LABEL_13:
  v15 = v25;
  *(_DWORD *)(a1 + 8) = v10 + v13;
  if ( v15 != v27 )
    _libc_free((unsigned __int64)v15);
}
