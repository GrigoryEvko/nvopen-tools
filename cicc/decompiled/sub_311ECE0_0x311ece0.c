// Function: sub_311ECE0
// Address: 0x311ece0
//
__int64 __fastcall sub_311ECE0(__int64 a1, __int64 a2)
{
  unsigned __int64 **v2; // r9
  __int64 v3; // r8
  unsigned __int64 **v4; // r15
  unsigned __int64 v5; // r12
  _QWORD *v7; // rax
  _QWORD *v8; // r15
  _QWORD *v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *i; // r10
  unsigned __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r14
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  char *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v21; // [rsp+10h] [rbp-40h]
  __int64 *v22; // [rsp+10h] [rbp-40h]
  unsigned __int64 **src; // [rsp+18h] [rbp-38h]
  unsigned __int64 **srca; // [rsp+18h] [rbp-38h]

  v2 = (unsigned __int64 **)(a1 + 16);
  v3 = a2;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  if ( !*(_DWORD *)(a2 + 16) )
    goto LABEL_2;
  v7 = *(_QWORD **)(a2 + 8);
  v8 = &v7[9 * *(unsigned int *)(a2 + 24)];
  if ( v7 == v8 )
    goto LABEL_2;
  while ( 1 )
  {
    v9 = v7;
    if ( *v7 <= 0xFFFFFFFFFFFFFFFDLL )
      break;
    v7 += 9;
    if ( v8 == v7 )
      goto LABEL_2;
  }
  if ( v8 == v7 )
  {
LABEL_2:
    v4 = (unsigned __int64 **)(a1 + 16);
LABEL_3:
    sub_311E6B0(v2, v4, v3);
    v5 = 0;
    goto LABEL_4;
  }
  v10 = 0;
  do
  {
    v11 = (__int64 *)v9[1];
    for ( i = &v11[*((unsigned int *)v9 + 4)]; i != v11; ++v11 )
    {
      v13 = *(unsigned int *)(a1 + 12);
      v14 = (unsigned int)v10;
      v15 = *v11;
      if ( (unsigned int)v10 >= v13 )
      {
        if ( v13 < (unsigned __int64)(unsigned int)v10 + 1 )
        {
          v20 = v3;
          v22 = i;
          srca = v2;
          sub_C8D5F0(a1, v2, (unsigned int)v10 + 1LL, 8u, v3, (__int64)v2);
          v14 = *(unsigned int *)(a1 + 8);
          v3 = v20;
          i = v22;
          v2 = srca;
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v14) = v15;
        v10 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
        *(_DWORD *)(a1 + 8) = v10;
      }
      else
      {
        v16 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)v10);
        if ( v16 )
        {
          *v16 = v15;
          LODWORD(v10) = *(_DWORD *)(a1 + 8);
        }
        v10 = (unsigned int)(v10 + 1);
        *(_DWORD *)(a1 + 8) = v10;
      }
    }
    v9 += 9;
    if ( v9 == v8 )
      break;
    while ( *v9 > 0xFFFFFFFFFFFFFFFDLL )
    {
      v9 += 9;
      if ( v8 == v9 )
        goto LABEL_20;
    }
  }
  while ( v8 != v9 );
LABEL_20:
  v17 = 8 * v10;
  v2 = *(unsigned __int64 ***)a1;
  v4 = (unsigned __int64 **)(*(_QWORD *)a1 + v17);
  v18 = v17 >> 3;
  if ( !v17 )
    goto LABEL_3;
  while ( 1 )
  {
    v21 = (unsigned __int64 *)v3;
    src = v2;
    v19 = (char *)sub_2207800(8 * v18);
    v2 = src;
    v3 = (__int64)v21;
    v5 = (unsigned __int64)v19;
    if ( v19 )
      break;
    v18 >>= 1;
    if ( !v18 )
      goto LABEL_3;
  }
  sub_311EBE0(src, v4, v19, v18, v21);
LABEL_4:
  j_j___libc_free_0(v5);
  return a1;
}
