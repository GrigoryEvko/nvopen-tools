// Function: sub_37234B0
// Address: 0x37234b0
//
void __fastcall sub_37234B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  _BYTE *v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // r14
  size_t v14; // rsi
  int v15; // r13d
  unsigned __int64 v16; // r8
  char *v17; // r8
  char *v18; // rax
  int v19; // esi
  char *v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // eax
  _DWORD *v23; // rcx
  void *base; // [rsp+0h] [rbp-40h] BYREF
  __int64 v25; // [rsp+8h] [rbp-38h]
  _BYTE v26[48]; // [rsp+10h] [rbp-30h] BYREF

  v7 = *(unsigned int *)(a1 + 136);
  base = v26;
  v8 = v26;
  v25 = 0;
  if ( !(_DWORD)v7 )
    goto LABEL_2;
  sub_C8D5F0((__int64)&base, v26, v7, 4u, a5, a6);
  v12 = *(_QWORD *)(a1 + 128);
  v13 = v12 + ((unsigned __int64)*(unsigned int *)(a1 + 136) << 6);
  if ( v13 == v12 )
  {
    v14 = (unsigned int)v25;
    v8 = base;
    v16 = 4LL * (unsigned int)v25;
  }
  else
  {
    v14 = (unsigned int)v25;
    do
    {
      v15 = *(_DWORD *)(v12 + 24);
      if ( v14 + 1 > HIDWORD(v25) )
      {
        sub_C8D5F0((__int64)&base, v26, v14 + 1, 4u, v10, v11);
        v14 = (unsigned int)v25;
      }
      v12 += 64;
      *((_DWORD *)base + v14) = v15;
      v14 = (unsigned int)(v25 + 1);
      LODWORD(v25) = v25 + 1;
    }
    while ( v13 != v12 );
    v8 = base;
    v16 = 4 * v14;
  }
  if ( v16 > 4 )
  {
    qsort(v8, v14, 4u, (__compar_fn_t)sub_2E1D450);
    v8 = base;
    v16 = 4LL * (unsigned int)v25;
  }
  v17 = &v8[v16];
  if ( v17 == v8 )
  {
LABEL_2:
    *(_DWORD *)(a1 + 156) = 0;
    v9 = 0;
  }
  else
  {
    v18 = v8;
    while ( 1 )
    {
      v20 = v18;
      v18 += 4;
      if ( v17 == v18 )
        break;
      v19 = *((_DWORD *)v18 - 1);
      if ( v19 == *(_DWORD *)v18 )
      {
        if ( v17 == v20 )
        {
          v18 = v17;
        }
        else
        {
          v23 = v20 + 8;
          if ( v20 + 8 != v17 )
          {
            while ( 1 )
            {
              if ( *v23 != v19 )
              {
                *((_DWORD *)v20 + 1) = *v23;
                v20 += 4;
              }
              if ( v17 == (char *)++v23 )
                break;
              v19 = *(_DWORD *)v20;
            }
            v8 = base;
            v18 = v20 + 4;
          }
        }
        break;
      }
    }
    v21 = (v18 - v8) >> 2;
    *(_DWORD *)(a1 + 156) = v21;
    v9 = v21;
    if ( (unsigned int)v21 > 0x400 )
    {
      v22 = (unsigned int)v21 >> 2;
      goto LABEL_19;
    }
    if ( (unsigned int)v21 > 0x10 )
    {
      v22 = (unsigned int)v21 >> 1;
      goto LABEL_19;
    }
  }
  v22 = 1;
  if ( v9 )
    v22 = v9;
LABEL_19:
  *(_DWORD *)(a1 + 152) = v22;
  if ( v8 != v26 )
    _libc_free((unsigned __int64)v8);
}
