// Function: sub_1FDEA40
// Address: 0x1fdea40
//
__int64 __fastcall sub_1FDEA40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v4; // rdi
  unsigned int v5; // ecx
  unsigned int *v6; // rdx
  __int64 v7; // r8
  int v9; // edx
  int v10; // r10d

  v2 = *(unsigned int *)(a1 + 392);
  v3 = 0x7FFFFFFF;
  if ( (_DWORD)v2 )
  {
    v4 = *(_QWORD *)(a1 + 376);
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (unsigned int *)(v4 + 16LL * v5);
    v7 = *(_QWORD *)v6;
    if ( a2 == *(_QWORD *)v6 )
    {
LABEL_3:
      if ( v6 != (unsigned int *)(v4 + 16 * v2) )
        return v6[2];
    }
    else
    {
      v9 = 1;
      while ( v7 != -8 )
      {
        v10 = v9 + 1;
        v5 = (v2 - 1) & (v9 + v5);
        v6 = (unsigned int *)(v4 + 16LL * v5);
        v7 = *(_QWORD *)v6;
        if ( a2 == *(_QWORD *)v6 )
          goto LABEL_3;
        v9 = v10;
      }
    }
    return 0x7FFFFFFF;
  }
  return v3;
}
