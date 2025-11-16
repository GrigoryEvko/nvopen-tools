// Function: sub_2E9EF00
// Address: 0x2e9ef00
//
unsigned int *__fastcall sub_2E9EF00(__int64 a1, __int64 a2, char a3)
{
  char v4; // cl
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned int *v7; // rdx
  unsigned int *result; // rax
  unsigned int *v9; // r9
  int *v10; // rsi
  int v11; // ecx
  unsigned int **v12; // rax
  __int64 v13; // rsi
  _BYTE v14[8]; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-38h]
  unsigned int *v16; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-28h]
  char v18; // [rsp+30h] [rbp-10h] BYREF

  sub_2E9CC30((__int64)v14, a1, a2, 1, a3);
  if ( v15 >> 1 )
  {
    v4 = v15 & 1;
    if ( (v15 & 1) != 0 )
    {
      v7 = (unsigned int *)&v18;
      result = (unsigned int *)&v16;
    }
    else
    {
      v5 = (__int64)v16;
      v6 = v17;
      v7 = &v16[2 * v17];
      result = v16;
      if ( v16 == v7 )
        goto LABEL_8;
    }
    do
    {
      if ( *result <= 0xFFFFFFFD )
        break;
      result += 2;
    }
    while ( v7 != result );
  }
  else
  {
    v4 = v15 & 1;
    if ( (v15 & 1) != 0 )
    {
      v13 = 4;
      v12 = &v16;
    }
    else
    {
      v12 = (unsigned int **)v16;
      v13 = v17;
    }
    result = (unsigned int *)&v12[v13];
    v7 = result;
  }
  if ( v4 )
  {
    v9 = (unsigned int *)&v18;
    if ( result == (unsigned int *)&v18 )
      return result;
    goto LABEL_9;
  }
  v5 = (__int64)v16;
  v6 = v17;
LABEL_8:
  v9 = (unsigned int *)(v5 + 8 * v6);
  if ( result == v9 )
    return (unsigned int *)sub_C7D6A0(v5, 8 * v6, 4);
  do
  {
LABEL_9:
    v10 = (int *)(*(_QWORD *)(a1 + 544) + 4LL * *result);
    v11 = *v10 + result[1];
    if ( *v10 < (signed int)-result[1] )
      v11 = 0;
    *v10 = v11;
    do
      result += 2;
    while ( result != v7 && *result > 0xFFFFFFFD );
  }
  while ( result != v9 );
  if ( (v15 & 1) == 0 )
  {
    v5 = (__int64)v16;
    v6 = v17;
    return (unsigned int *)sub_C7D6A0(v5, 8 * v6, 4);
  }
  return result;
}
