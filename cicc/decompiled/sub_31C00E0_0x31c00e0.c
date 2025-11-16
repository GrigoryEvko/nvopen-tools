// Function: sub_31C00E0
// Address: 0x31c00e0
//
void __fastcall sub_31C00E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  int v4; // edx
  unsigned int v6; // eax
  __int64 *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 v9; // r13
  __int64 *v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  int v13; // r8d

  v2 = *(_DWORD *)(a1 + 240);
  v3 = *(_QWORD *)(a1 + 224);
  if ( v2 )
  {
    v4 = v2 - 1;
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v9 = v7[1];
      if ( v9 )
      {
        v10 = *(__int64 **)v9;
        v11 = *(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8);
        if ( *(_QWORD *)v9 != v11 )
        {
          do
          {
            v12 = *v10++;
            *(_QWORD *)(v12 + 32) = 0;
          }
          while ( (__int64 *)v11 != v10 );
          v11 = *(_QWORD *)v9;
        }
        if ( v11 != v9 + 16 )
          _libc_free(v11);
        j_j___libc_free_0(v9);
      }
      *v7 = -8192;
      --*(_DWORD *)(a1 + 232);
      ++*(_DWORD *)(a1 + 236);
    }
    else
    {
      v13 = 1;
      while ( v8 != -4096 )
      {
        v6 = v4 & (v13 + v6);
        v7 = (__int64 *)(v3 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        ++v13;
      }
    }
  }
}
