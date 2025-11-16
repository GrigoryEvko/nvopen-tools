// Function: sub_2B42D90
// Address: 0x2b42d90
//
unsigned __int64 __fastcall sub_2B42D90(__int64 a1, unsigned int a2, const void *a3, unsigned __int64 a4)
{
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rbx
  unsigned __int64 result; // rax
  __int64 i; // rdx
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // eax
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // rax
  unsigned __int64 *v24; // r14
  int v25; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-50h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  unsigned int v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a2 + 1;
  v8 = *(unsigned int *)(a1 + 248);
  if ( (unsigned int)v7 <= (unsigned int)v8 || (unsigned int)v7 == v8 )
  {
    v9 = *(_QWORD *)(a1 + 240);
    goto LABEL_3;
  }
  v14 = 80LL * (unsigned int)v7;
  if ( (unsigned int)v7 < v8 )
  {
    v9 = *(_QWORD *)(a1 + 240);
    v23 = (unsigned __int64 *)(v9 + 80 * v8);
    v24 = (unsigned __int64 *)(v9 + v14);
    if ( v23 == v24 )
      goto LABEL_23;
    do
    {
      v23 -= 10;
      if ( (unsigned __int64 *)*v23 != v23 + 2 )
      {
        v26 = v23;
        v28 = v7;
        _libc_free(*v23);
        v23 = v26;
        v7 = v28;
      }
    }
    while ( v24 != v23 );
  }
  else
  {
    if ( (unsigned int)v7 > (unsigned __int64)*(unsigned int *)(a1 + 252) )
    {
      v27 = sub_C8D7D0(a1 + 240, a1 + 256, (unsigned int)v7, 0x50u, v29, v7);
      sub_2B42CC0(a1 + 240, v27, v17, v18, v19, v20);
      v21 = v29[0];
      v22 = *(_QWORD *)(a1 + 240);
      v9 = v27;
      v7 = a2 + 1;
      if ( a1 + 256 != v22 )
      {
        v25 = v29[0];
        _libc_free(v22);
        v9 = v27;
        v21 = v25;
        v7 = a2 + 1;
      }
      *(_DWORD *)(a1 + 252) = v21;
      v8 = *(unsigned int *)(a1 + 248);
      *(_QWORD *)(a1 + 240) = v9;
    }
    else
    {
      v9 = *(_QWORD *)(a1 + 240);
    }
    v15 = v9 + v14;
    v16 = v9 + 80 * v8;
    if ( v16 == v15 )
      goto LABEL_23;
    do
    {
      if ( v16 )
      {
        *(_DWORD *)(v16 + 8) = 0;
        *(_QWORD *)v16 = v16 + 16;
        *(_DWORD *)(v16 + 12) = 8;
      }
      v16 += 80;
    }
    while ( v15 != v16 );
  }
  v9 = *(_QWORD *)(a1 + 240);
LABEL_23:
  *(_DWORD *)(a1 + 248) = v7;
LABEL_3:
  v10 = 8 * a4;
  v11 = v9 + 80LL * a2;
  result = *(unsigned int *)(v11 + 8);
  if ( a4 == result )
    goto LABEL_11;
  if ( a4 >= result )
  {
    if ( a4 > *(unsigned int *)(v11 + 12) )
    {
      sub_C8D5F0(v11, (const void *)(v11 + 16), a4, 8u, v10, v7);
      v10 = 8 * a4;
      result = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
      for ( i = 8 * a4 + *(_QWORD *)v11; i != result; result += 8LL )
      {
LABEL_7:
        if ( result )
          *(_QWORD *)result = 0;
      }
    }
    else
    {
      result = *(_QWORD *)v11 + 8 * result;
      i = v10 + *(_QWORD *)v11;
      if ( result != i )
        goto LABEL_7;
    }
  }
  *(_DWORD *)(v11 + 8) = a4;
  v11 = 80LL * a2 + *(_QWORD *)(a1 + 240);
LABEL_11:
  if ( v10 )
    return (unsigned __int64)memmove(*(void **)v11, a3, v10);
  return result;
}
