// Function: sub_37B6270
// Address: 0x37b6270
//
unsigned int *__fastcall sub_37B6270(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 i; // rdx
  __int64 v11; // rcx
  unsigned int *result; // rax
  unsigned int v13; // r13d
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx

  v8 = a3 & 1;
  v9 = (a3 - 1) / 2;
  if ( a2 >= v9 )
  {
    result = (unsigned int *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v11 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v11 )
  {
    v11 = 2 * (i + 1);
    result = (unsigned int *)(a1 + 32 * (i + 1));
    v13 = *result;
    if ( *result < *(result - 4) )
    {
      --v11;
      result = (unsigned int *)(a1 + 16 * v11);
      v13 = *result;
    }
    v14 = a1 + 16 * i;
    *(_DWORD *)v14 = v13;
    *(_QWORD *)(v14 + 8) = *((_QWORD *)result + 1);
    if ( v11 >= v9 )
      break;
  }
  if ( !v8 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v11 )
    {
      v17 = v11 + 1;
      v11 = 2 * (v11 + 1) - 1;
      v18 = a1 + 32 * v17 - 16;
      *result = *(_DWORD *)v18;
      *((_QWORD *)result + 1) = *(_QWORD *)(v18 + 8);
      result = (unsigned int *)(a1 + 16 * v11);
    }
  }
  v15 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    while ( 1 )
    {
      result = (unsigned int *)(a1 + 16 * v11);
      v16 = (unsigned int *)(a1 + 16 * v15);
      if ( a4 <= *v16 )
        break;
      *result = *v16;
      *((_QWORD *)result + 1) = *((_QWORD *)v16 + 1);
      v11 = v15;
      if ( a2 >= v15 )
      {
        *v16 = a4;
        *((_QWORD *)v16 + 1) = a5;
        return (unsigned int *)(a1 + 16 * v15);
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_13:
  *result = a4;
  *((_QWORD *)result + 1) = a5;
  return result;
}
