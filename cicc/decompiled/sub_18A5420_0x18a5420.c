// Function: sub_18A5420
// Address: 0x18a5420
//
int *__fastcall sub_18A5420(int *a1, size_t a2, int a3, _QWORD *a4)
{
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // eax
  _BYTE *v12; // rsi
  int v13; // ecx
  __int64 v14; // r10
  unsigned __int64 v15; // rdx
  char v16; // r11
  __int64 v17; // rax
  _BYTE *v18; // rdi
  size_t v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int64 v23; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD *v24; // [rsp+20h] [rbp-D0h] BYREF
  size_t n; // [rsp+28h] [rbp-C8h]
  _QWORD v26[24]; // [rsp+30h] [rbp-C0h] BYREF

  if ( !a2 )
    return a1;
  sub_16C1840(&v24);
  sub_16C1A90((int *)&v24, a1, a2);
  sub_16C1AA0(&v24, &v23);
  v8 = v23;
  if ( v23 <= 9 )
  {
    v24 = v26;
    sub_2240A50(&v24, 1, 0, v7, v26);
    v12 = v24;
LABEL_16:
    *v12 = v8 + 48;
    goto LABEL_17;
  }
  if ( v23 <= 0x63 )
  {
    v24 = v26;
    sub_2240A50(&v24, 2, 0, v7, v26);
    v12 = v24;
  }
  else
  {
    if ( v23 <= 0x3E7 )
    {
      v10 = 3;
    }
    else if ( v23 <= 0x270F )
    {
      v10 = 4;
    }
    else
    {
      v9 = v23;
      LODWORD(v10) = 1;
      while ( 1 )
      {
        v7 = v9;
        v11 = v10;
        v10 = (unsigned int)(v10 + 4);
        v9 /= 0x2710u;
        if ( v7 <= 0x1869F )
          break;
        if ( v7 <= 0xF423F )
        {
          v10 = (unsigned int)(v11 + 5);
          v24 = v26;
          goto LABEL_13;
        }
        if ( v7 <= (unsigned __int64)&loc_98967F )
        {
          v10 = (unsigned int)(v11 + 6);
          break;
        }
        if ( v7 <= 0x5F5E0FF )
        {
          v10 = (unsigned int)(v11 + 7);
          break;
        }
      }
    }
    v24 = v26;
LABEL_13:
    sub_2240A50(&v24, v10, 0, v7, v26);
    v12 = v24;
    v13 = n - 1;
    do
    {
      v14 = v8
          - 20 * (v8 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v8 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v15 = v8;
      v8 /= 0x64u;
      v16 = a00010203040506_0[2 * v14 + 1];
      LOBYTE(v14) = a00010203040506_0[2 * v14];
      v12[v13] = v16;
      v17 = (unsigned int)(v13 - 1);
      v13 -= 2;
      v12[v17] = v14;
    }
    while ( v15 > 0x270F );
    if ( v15 <= 0x3E7 )
      goto LABEL_16;
  }
  v12[1] = a00010203040506_0[2 * v8 + 1];
  *v12 = a00010203040506_0[2 * v8];
LABEL_17:
  v18 = (_BYTE *)*a4;
  v19 = n;
  if ( v24 == v26 )
  {
    if ( n )
    {
      if ( n == 1 )
        *v18 = v26[0];
      else
        memcpy(v18, v26, n);
      v19 = n;
      v18 = (_BYTE *)*a4;
    }
    a4[1] = v19;
    v18[v19] = 0;
    v18 = v24;
  }
  else
  {
    v20 = v26[0];
    if ( v18 == (_BYTE *)(a4 + 2) )
    {
      *a4 = v24;
      a4[1] = v19;
      a4[2] = v20;
    }
    else
    {
      v21 = a4[2];
      *a4 = v24;
      a4[1] = v19;
      a4[2] = v20;
      if ( v18 )
      {
        v24 = v18;
        v26[0] = v21;
        goto LABEL_21;
      }
    }
    v24 = v26;
    v18 = v26;
  }
LABEL_21:
  n = 0;
  *v18 = 0;
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0] + 1LL);
  if ( a3 != 2 )
    return a1;
  return (int *)*a4;
}
