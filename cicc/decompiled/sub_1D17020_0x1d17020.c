// Function: sub_1D17020
// Address: 0x1d17020
//
void __fastcall sub_1D17020(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // r9d
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 *v7; // r12
  __int64 v8; // rdi
  __int64 *v9; // rdi
  __int64 *v10; // rdx
  __int64 v11; // rax

  v2 = *(unsigned int *)(a1 + 720);
  if ( (_DWORD)v2 )
  {
    v3 = 1;
    v5 = *(_QWORD *)(a1 + 704);
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 40LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 40 * v2) )
      {
        v9 = (__int64 *)v7[1];
        v10 = &v9[*((unsigned int *)v7 + 4)];
        if ( v10 != v9 )
        {
          do
          {
            v11 = *v9++;
            *(_BYTE *)(v11 + 49) = 1;
          }
          while ( v10 != v9 );
          v9 = (__int64 *)v7[1];
        }
        if ( v9 != v7 + 3 )
          _libc_free((unsigned __int64)v9);
        *v7 = -16;
        --*(_DWORD *)(a1 + 712);
        ++*(_DWORD *)(a1 + 716);
      }
    }
    else
    {
      while ( v8 != -8 )
      {
        v6 = (v2 - 1) & (v3 + v6);
        v7 = (__int64 *)(v5 + 40LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v3;
      }
    }
  }
}
