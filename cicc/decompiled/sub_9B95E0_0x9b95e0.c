// Function: sub_9B95E0
// Address: 0x9b95e0
//
__int64 *__fastcall sub_9B95E0(__int64 *a1, int a2, int a3, int a4)
{
  __int64 *v7; // rdx
  int v8; // r13d
  __int64 v9; // rax

  *a1 = (__int64)(a1 + 2);
  a1[1] = 0x1000000000LL;
  if ( a4 )
  {
    v7 = a1 + 2;
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
      *((_DWORD *)v7 + v9) = a2;
      ++v8;
      v9 = (unsigned int)(*((_DWORD *)a1 + 2) + 1);
      *((_DWORD *)a1 + 2) = v9;
      if ( a4 == v8 )
        break;
      if ( v9 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        sub_C8D5F0(a1, a1 + 2, v9 + 1, 4);
        v9 = *((unsigned int *)a1 + 2);
      }
      v7 = (__int64 *)*a1;
      a2 += a3;
    }
  }
  return a1;
}
