// Function: sub_D97100
// Address: 0xd97100
//
__int64 __fastcall sub_D97100(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 *v5; // rcx
  __int64 v6; // rdi
  int v8; // ecx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 120);
  v3 = *(_QWORD *)(a1 + 104);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 88LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
    {
LABEL_3:
      if ( v5 != (__int64 *)(v3 + 88 * v2) )
        return v5[5];
    }
    else
    {
      v8 = 1;
      while ( v6 != -4096 )
      {
        v9 = v8 + 1;
        v4 = (v2 - 1) & (v8 + v4);
        v5 = (__int64 *)(v3 + 88LL * v4);
        v6 = *v5;
        if ( *v5 == a2 )
          goto LABEL_3;
        v8 = v9;
      }
    }
  }
  return 0;
}
