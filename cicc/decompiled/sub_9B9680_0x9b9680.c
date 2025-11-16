// Function: sub_9B9680
// Address: 0x9b9680
//
__int64 *__fastcall sub_9B9680(__int64 *a1, int a2, int a3, int a4)
{
  __int64 *v4; // r14
  int v6; // r15d
  int v7; // ebx
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // ebx

  v4 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  a1[1] = 0x1000000000LL;
  if ( a3 )
  {
    v6 = a2 + a3 - 1;
    v7 = a2;
    v8 = a1 + 2;
    v9 = 0;
    while ( 1 )
    {
      *((_DWORD *)v8 + v9) = v7;
      v9 = (unsigned int)(*((_DWORD *)a1 + 2) + 1);
      *((_DWORD *)a1 + 2) = v9;
      if ( v7 == v6 )
        break;
      if ( v9 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        sub_C8D5F0(a1, v4, v9 + 1, 4);
        v9 = *((unsigned int *)a1 + 2);
      }
      v8 = (__int64 *)*a1;
      ++v7;
    }
  }
  if ( a4 )
  {
    v10 = *((unsigned int *)a1 + 2);
    v11 = 0;
    do
    {
      if ( v10 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        sub_C8D5F0(a1, v4, v10 + 1, 4);
        v10 = *((unsigned int *)a1 + 2);
      }
      ++v11;
      *(_DWORD *)(*a1 + 4 * v10) = -1;
      v10 = (unsigned int)(*((_DWORD *)a1 + 2) + 1);
      *((_DWORD *)a1 + 2) = v10;
    }
    while ( a4 != v11 );
  }
  return a1;
}
