// Function: sub_E10300
// Address: 0xe10300
//
__int64 __fastcall sub_E10300(__int64 a1, __int64 *a2)
{
  char *v2; // r13
  unsigned __int64 v4; // rdx
  char *v5; // rax
  __int64 v6; // rbx
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  char *v11; // rdi
  __int64 result; // rax
  unsigned __int64 v13; // rcx
  char *v14; // rdi
  char *v15; // r13
  unsigned int v16; // edx
  __int64 v17; // rsi

  v2 = "false";
  v4 = a2[2];
  if ( *(_BYTE *)(a1 + 11) )
    v2 = "true";
  v5 = (char *)a2[1];
  v6 = (*(_BYTE *)(a1 + 11) == 0) + 4LL;
  v7 = (char *)*a2;
  if ( (unsigned __int64)&v5[v6] > v4 )
  {
    v8 = (unsigned __int64)&v5[v6 + 992];
    v9 = 2 * v4;
    if ( v8 > v9 )
      a2[2] = v8;
    else
      a2[2] = v9;
    v10 = realloc(v7);
    *a2 = v10;
    v7 = (char *)v10;
    if ( !v10 )
      abort();
    v5 = (char *)a2[1];
  }
  v11 = &v7[(_QWORD)v5];
  result = (unsigned int)v6;
  if ( (unsigned int)v6 < 8 )
  {
    if ( (v6 & 4) != 0 )
    {
      *(_DWORD *)v11 = *(_DWORD *)v2;
      *(_DWORD *)&v11[(unsigned int)v6 - 4] = *(_DWORD *)&v2[(unsigned int)v6 - 4];
    }
    else if ( (_DWORD)v6 )
    {
      *v11 = *v2;
      if ( (v6 & 2) != 0 )
        *(_WORD *)&v11[(unsigned int)v6 - 2] = *(_WORD *)&v2[(unsigned int)v6 - 2];
    }
    goto LABEL_13;
  }
  v13 = (unsigned __int64)(v11 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = *(_QWORD *)v2;
  *(_QWORD *)&v11[(unsigned int)v6 - 8] = *(_QWORD *)&v2[(unsigned int)v6 - 8];
  v14 = &v11[-v13];
  v15 = (char *)(v2 - v14);
  result = ((_DWORD)v6 + (_DWORD)v14) & 0xFFFFFFF8;
  if ( (unsigned int)result < 8 )
  {
LABEL_13:
    a2[1] += v6;
    return result;
  }
  result = ((_DWORD)v6 + (_DWORD)v14) & 0xFFFFFFF8;
  v16 = 0;
  do
  {
    v17 = v16;
    v16 += 8;
    *(_QWORD *)(v13 + v17) = *(_QWORD *)&v15[v17];
  }
  while ( v16 < (unsigned int)result );
  a2[1] += v6;
  return result;
}
