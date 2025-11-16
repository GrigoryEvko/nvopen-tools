// Function: sub_A6A250
// Address: 0xa6a250
//
__int64 __fastcall sub_A6A250(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // edx
  __int64 *v6; // rcx
  __int64 v7; // rdi
  int v9; // ecx
  int v10; // r9d

  sub_A6A190(a1);
  v3 = *(unsigned int *)(a1 + 320);
  v4 = *(_QWORD *)(a1 + 304);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
        return *((unsigned int *)v6 + 2);
    }
    else
    {
      v9 = 1;
      while ( v7 != -1 )
      {
        v10 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  return 0xFFFFFFFFLL;
}
