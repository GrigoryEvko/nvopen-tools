// Function: sub_CFBB40
// Address: 0xcfbb40
//
__int64 __fastcall sub_CFBB40(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rdx
  __int64 v3; // rax
  unsigned __int64 *v4; // r13
  __int64 v5; // rcx
  __int64 result; // rax
  unsigned __int64 *v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r14
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned __int64 **)a1;
  v3 = 32LL * *(unsigned int *)(a1 + 8);
  v4 = (unsigned __int64 *)(*(_QWORD *)a1 + v3);
  v5 = v3 >> 5;
  result = v3 >> 7;
  if ( result )
  {
    v7 = *(unsigned __int64 **)a1;
    result = (__int64)&v2[16 * result];
    while ( v7[2] != a2 )
    {
      if ( v7[6] == a2 )
      {
        v7 += 4;
        goto LABEL_8;
      }
      if ( v7[10] == a2 )
      {
        v7 += 8;
        goto LABEL_8;
      }
      if ( v7[14] == a2 )
      {
        v7 += 12;
        goto LABEL_8;
      }
      v7 += 16;
      if ( (unsigned __int64 *)result == v7 )
      {
        v5 = ((char *)v4 - (char *)v7) >> 5;
        goto LABEL_42;
      }
    }
    goto LABEL_8;
  }
  v7 = *(unsigned __int64 **)a1;
LABEL_42:
  if ( v5 == 2 )
    goto LABEL_49;
  if ( v5 == 3 )
  {
    if ( v7[2] == a2 )
      goto LABEL_8;
    v7 += 4;
LABEL_49:
    if ( v7[2] == a2 )
      goto LABEL_8;
    v7 += 4;
    goto LABEL_45;
  }
  if ( v5 != 1 )
    goto LABEL_37;
LABEL_45:
  if ( v7[2] != a2 )
    goto LABEL_37;
LABEL_8:
  if ( v4 == v7 )
    goto LABEL_37;
  v8 = (__int64)(v7 + 4);
  if ( v4 == v7 + 4 )
  {
    v4 = v7;
    do
    {
LABEL_32:
      result = *(_QWORD *)(v8 - 16);
      v8 -= 32;
      if ( result != 0 && result != -4096 && result != -8192 )
        result = sub_BD60C0((_QWORD *)v8);
    }
    while ( (unsigned __int64 *)v8 != v4 );
    v2 = *(unsigned __int64 **)a1;
    goto LABEL_37;
  }
  do
  {
    v9 = *(_QWORD *)(v8 + 16);
    if ( v9 != a2 )
    {
      v10 = v7[2];
      if ( v9 != v10 )
      {
        if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        {
          sub_BD60C0(v7);
          v9 = *(_QWORD *)(v8 + 16);
        }
        v7[2] = v9;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
          sub_BD6050(v7, *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v7 += 4;
      *((_DWORD *)v7 - 2) = *(_DWORD *)(v8 + 24);
    }
    v8 += 32;
  }
  while ( v4 != (unsigned __int64 *)v8 );
  v2 = *(unsigned __int64 **)a1;
  v8 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  v16 = v8 - (_QWORD)v4;
  result = v8 - (_QWORD)v4;
  v11 = (v8 - (__int64)v4) >> 5;
  if ( v8 - (__int64)v4 <= 0 )
  {
    v4 = v7;
  }
  else
  {
    v12 = v7;
    do
    {
      v13 = v12[2];
      v14 = v4[2];
      if ( v13 != v14 )
      {
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
        {
          sub_BD60C0(v12);
          v14 = v4[2];
        }
        v12[2] = v14;
        if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
          sub_BD6050(v12, *v4 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v15 = *((_DWORD *)v4 + 6);
      v12 += 4;
      v4 += 4;
      *((_DWORD *)v12 - 2) = v15;
      --v11;
    }
    while ( v11 );
    result = v16;
    v2 = *(unsigned __int64 **)a1;
    v8 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    v4 = (unsigned __int64 *)((char *)v7 + v16);
  }
  if ( v4 != (unsigned __int64 *)v8 )
    goto LABEL_32;
LABEL_37:
  *(_DWORD *)(a1 + 8) = ((char *)v4 - (char *)v2) >> 5;
  return result;
}
