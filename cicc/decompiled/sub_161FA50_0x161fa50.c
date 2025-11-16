// Function: sub_161FA50
// Address: 0x161fa50
//
void __fastcall sub_161FA50(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  int v4; // eax
  int v5; // edx
  __int64 v6; // rdi
  unsigned int v7; // eax
  int v8; // esi
  __int64 *v9; // r14
  __int64 v10; // rcx
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  __int64 v13; // rsi

  v2 = sub_16498A0(a1);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(*(_QWORD *)v2 + 2728LL);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(v3 + 2712);
    v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = 1;
    v9 = (__int64 *)(v6 + 56LL * v7);
    v10 = *v9;
    if ( a1 == *v9 )
    {
LABEL_3:
      v11 = v9[1];
      v12 = v11 + 16LL * *((unsigned int *)v9 + 4);
      if ( v11 != v12 )
      {
        do
        {
          v13 = *(_QWORD *)(v12 - 8);
          v12 -= 16LL;
          if ( v13 )
            sub_161E7C0(v12 + 8, v13);
        }
        while ( v11 != v12 );
        v12 = v9[1];
      }
      if ( (__int64 *)v12 != v9 + 3 )
        _libc_free(v12);
      *v9 = -16;
      --*(_DWORD *)(v3 + 2720);
      ++*(_DWORD *)(v3 + 2724);
    }
    else
    {
      while ( v10 != -8 )
      {
        v7 = v5 & (v8 + v7);
        v9 = (__int64 *)(v6 + 56LL * v7);
        v10 = *v9;
        if ( a1 == *v9 )
          goto LABEL_3;
        ++v8;
      }
    }
  }
  *(_WORD *)(a1 + 18) &= ~0x8000u;
}
