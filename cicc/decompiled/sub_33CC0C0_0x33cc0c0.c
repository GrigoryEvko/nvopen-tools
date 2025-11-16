// Function: sub_33CC0C0
// Address: 0x33cc0c0
//
void __fastcall sub_33CC0C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned int v5; // edx
  __int64 *v6; // r12
  __int64 v7; // rcx
  __int64 *v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // rax
  int v11; // r9d

  v3 = *(_QWORD *)(a1 + 696);
  v4 = *(unsigned int *)(a1 + 712);
  if ( (_DWORD)v4 )
  {
    v5 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v3 + 40LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v3 + 40 * v4) )
      {
        v8 = (__int64 *)v6[1];
        v9 = &v8[*((unsigned int *)v6 + 4)];
        if ( v9 != v8 )
        {
          do
          {
            v10 = *v8++;
            *(_BYTE *)(v10 + 62) = 1;
          }
          while ( v8 != v9 );
          v8 = (__int64 *)v6[1];
        }
        if ( v8 != v6 + 3 )
          _libc_free((unsigned __int64)v8);
        *v6 = -8192;
        --*(_DWORD *)(a1 + 704);
        ++*(_DWORD *)(a1 + 708);
      }
    }
    else
    {
      v11 = 1;
      while ( v7 != -4096 )
      {
        v5 = (v4 - 1) & (v11 + v5);
        v6 = (__int64 *)(v3 + 40LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        ++v11;
      }
    }
  }
}
