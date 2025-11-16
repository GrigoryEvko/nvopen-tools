// Function: sub_2FACCA0
// Address: 0x2facca0
//
void __fastcall sub_2FACCA0(__int64 *a1, __int64 *a2)
{
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx

  if ( a1 != a2 )
  {
    v4 = a1 + 2;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *v4;
        v6 = v4;
        v4 += 2;
        if ( (*(_DWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v5 >> 1) & 3) >= (*(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(*a1 >> 1)
                                                                                             & 3) )
          break;
        v7 = *(v4 - 2);
        v8 = *(v4 - 1);
        v9 = ((char *)v6 - (char *)a1) >> 4;
        if ( (char *)v6 - (char *)a1 > 0 )
        {
          do
          {
            v10 = *(v6 - 2);
            v6 -= 2;
            v6[2] = v10;
            v6[3] = v6[1];
            --v9;
          }
          while ( v9 );
        }
        *a1 = v7;
        a1[1] = v8;
        if ( a2 == v4 )
          return;
      }
      sub_2FACC30(v6);
    }
  }
}
