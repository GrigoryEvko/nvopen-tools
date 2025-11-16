// Function: sub_2EAA350
// Address: 0x2eaa350
//
void __fastcall sub_2EAA350(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rcx
  void (*v5)(); // rdx
  unsigned int v6; // eax
  __int64 *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 v9; // r13
  int v10; // r8d

  v3 = *(_DWORD *)(a1 + 2528);
  v4 = *(_QWORD *)(a1 + 2512);
  if ( v3 )
  {
    v5 = (void (*)())(unsigned int)(v3 - 1);
    v6 = (unsigned int)v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v9 = v7[1];
      if ( v9 )
      {
        sub_2E81F20(v7[1], a2, v5, v4);
        j_j___libc_free_0(v9);
      }
      *v7 = -8192;
      --*(_DWORD *)(a1 + 2520);
      ++*(_DWORD *)(a1 + 2524);
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v6 = (unsigned int)v5 & (v10 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        ++v10;
      }
    }
  }
  *(_QWORD *)(a1 + 2544) = 0;
  *(_QWORD *)(a1 + 2552) = 0;
}
