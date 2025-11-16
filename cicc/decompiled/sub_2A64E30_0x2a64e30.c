// Function: sub_2A64E30
// Address: 0x2a64e30
//
void __fastcall sub_2A64E30(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // eax
  __int64 v4; // rdi
  int v5; // edx
  unsigned int v6; // eax
  __int64 *v7; // rbx
  __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  int v11; // r8d

  v2 = *a1;
  v3 = *(_DWORD *)(*a1 + 160);
  v4 = *(_QWORD *)(*a1 + 144);
  if ( v3 )
  {
    v5 = v3 - 1;
    v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 48LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( (unsigned int)*((unsigned __int8 *)v7 + 8) - 4 <= 1 )
      {
        if ( *((_DWORD *)v7 + 10) > 0x40u )
        {
          v9 = v7[4];
          if ( v9 )
            j_j___libc_free_0_0(v9);
        }
        if ( *((_DWORD *)v7 + 6) > 0x40u )
        {
          v10 = v7[2];
          if ( v10 )
            j_j___libc_free_0_0(v10);
        }
      }
      *v7 = -8192;
      --*(_DWORD *)(v2 + 152);
      ++*(_DWORD *)(v2 + 156);
    }
    else
    {
      v11 = 1;
      while ( v8 != -4096 )
      {
        v6 = v5 & (v11 + v6);
        v7 = (__int64 *)(v4 + 48LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v11;
      }
    }
  }
}
