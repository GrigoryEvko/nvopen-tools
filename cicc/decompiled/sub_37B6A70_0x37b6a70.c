// Function: sub_37B6A70
// Address: 0x37b6a70
//
__int64 __fastcall sub_37B6A70(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v7; // r14
  __int64 v9; // r12
  unsigned int *v10; // rdx
  unsigned int v11; // ecx
  __int64 v12; // rcx
  unsigned int *v13; // r10

  result = a3 - 1;
  v5 = a2;
  v7 = a3 & 1;
  v9 = (a3 - 1) / 2;
  if ( a2 >= v9 )
  {
    v10 = (unsigned int *)(a1 + 4 * a2);
    if ( v7 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    result = 2 * (a2 + 1) - 1;
    v10 = (unsigned int *)(a1 + 4 * result);
    v11 = *v10;
    if ( *(_DWORD *)(a1 + 8 * (a2 + 1)) <= *v10 )
    {
      v11 = *(_DWORD *)(a1 + 8 * (a2 + 1));
      v10 = (unsigned int *)(a1 + 8 * (a2 + 1));
      result = 2 * (a2 + 1);
    }
    *(_DWORD *)(a1 + 4 * a2) = v11;
    if ( result >= v9 )
      break;
    a2 = result;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      *v10 = *(_DWORD *)(a1 + 4 * result);
      v10 = (unsigned int *)(a1 + 4 * result);
    }
  }
  v12 = (result - 1) / 2;
  if ( result > v5 )
  {
    while ( 1 )
    {
      v13 = (unsigned int *)(a1 + 4 * v12);
      v10 = (unsigned int *)(a1 + 4 * result);
      if ( a4 >= *v13 )
        break;
      *v10 = *v13;
      result = v12;
      if ( v5 >= v12 )
      {
        *v13 = a4;
        return result;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = a4;
  return result;
}
