// Function: sub_2DB9B50
// Address: 0x2db9b50
//
__int64 __fastcall sub_2DB9B50(__int64 *a1)
{
  _DWORD *v1; // r15
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v4; // r12d
  __int64 *v5; // r14
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 j; // rcx
  int v18; // r12d
  __int64 result; // rax
  int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // rbx
  __int64 i; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+0h] [rbp-40h]
  unsigned int v28; // [rsp+0h] [rbp-40h]

  v1 = a1 + 1;
  v2 = *a1;
  *((_DWORD *)a1 + 4) = 0;
  *((_DWORD *)a1 + 14) = 0;
  sub_3157150(a1 + 1, 2 * (unsigned int)((__int64)(*(_QWORD *)(v2 + 104) - *(_QWORD *)(v2 + 96)) >> 3));
  v3 = *(_QWORD *)(*a1 + 328);
  for ( i = *a1 + 320; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    v4 = 2 * *(_DWORD *)(v3 + 24) + 1;
    v5 = *(__int64 **)(v3 + 112);
    v6 = &v5[*(unsigned int *)(v3 + 120)];
    while ( v6 != v5 )
    {
      v7 = *v5++;
      sub_31571F0(v1, v4, (unsigned int)(2 * *(_DWORD *)(v7 + 24)));
    }
  }
  sub_3157250(v1);
  if ( (_BYTE)qword_501D1C8 )
    sub_2DB9A90(a1);
  v11 = (unsigned __int64 *)a1[8];
  v12 = &v11[6 * *((unsigned int *)a1 + 18)];
  while ( v11 != v12 )
  {
    while ( 1 )
    {
      v12 -= 6;
      if ( (unsigned __int64 *)*v12 == v12 + 2 )
        break;
      _libc_free(*v12);
      if ( v11 == v12 )
        goto LABEL_11;
    }
  }
LABEL_11:
  v13 = *((unsigned int *)a1 + 14);
  *((_DWORD *)a1 + 18) = 0;
  if ( v13 )
  {
    v14 = *((unsigned int *)a1 + 19);
    v15 = 0;
    if ( v13 > v14 )
    {
      sub_239A310((__int64)(a1 + 8), v13, v14, v8, v9, v10);
      v15 = 48LL * *((unsigned int *)a1 + 18);
    }
    v27 = a1[8];
    v16 = v27 + v15;
    for ( j = v27 + 48 * v13; j != v16; v16 += 48 )
    {
      if ( v16 )
      {
        *(_DWORD *)(v16 + 8) = 0;
        *(_QWORD *)v16 = v16 + 16;
        *(_DWORD *)(v16 + 12) = 8;
      }
    }
    *((_DWORD *)a1 + 18) = v13;
  }
  v18 = 0;
  result = (__int64)(*(_QWORD *)(*a1 + 104) - *(_QWORD *)(*a1 + 96)) >> 3;
  v20 = result;
  if ( (_DWORD)result )
  {
    do
    {
      v21 = a1[1];
      v22 = *(unsigned int *)(v21 + 4LL * (unsigned int)(2 * v18));
      v23 = *(unsigned int *)(v21 + 4LL * (unsigned int)(2 * v18 + 1));
      v24 = a1[8] + 48 * v22;
      result = *(unsigned int *)(v24 + 8);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(v24 + 12) )
      {
        v28 = v22;
        sub_C8D5F0(v24, (const void *)(v24 + 16), result + 1, 4u, v22, v10);
        result = *(unsigned int *)(v24 + 8);
        v22 = v28;
      }
      *(_DWORD *)(*(_QWORD *)v24 + 4 * result) = v18;
      ++*(_DWORD *)(v24 + 8);
      if ( (_DWORD)v23 != (_DWORD)v22 )
      {
        v25 = a1[8] + 48 * v23;
        result = *(unsigned int *)(v25 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(v25 + 12) )
        {
          sub_C8D5F0(a1[8] + 48 * v23, (const void *)(v25 + 16), result + 1, 4u, v22, v10);
          result = *(unsigned int *)(v25 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v25 + 4 * result) = v18;
        ++*(_DWORD *)(v25 + 8);
      }
      ++v18;
    }
    while ( v18 != v20 );
  }
  return result;
}
