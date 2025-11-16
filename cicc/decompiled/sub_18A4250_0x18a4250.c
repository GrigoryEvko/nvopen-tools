// Function: sub_18A4250
// Address: 0x18a4250
//
__int64 __fastcall sub_18A4250(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 *v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // r14d
  __int64 i; // r12
  __int64 j; // r15
  int v13; // ecx
  int v14; // r10d

  v4 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 56LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 56 * v4) )
      {
        v9 = *((_DWORD *)v7 + 12);
        goto LABEL_5;
      }
    }
    else
    {
      v13 = 1;
      while ( v8 != -8 )
      {
        v14 = v13 + 1;
        v6 = (v4 - 1) & (v13 + v6);
        v7 = (__int64 *)(v5 + 56LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v13 = v14;
      }
    }
  }
  v9 = 0;
LABEL_5:
  for ( i = *(_QWORD *)(a2 + 104); a2 + 88 != i; i = sub_220EF30(i) )
  {
    for ( j = *(_QWORD *)(i + 64); i + 48 != j; j = sub_220EF30(j) )
    {
      if ( sub_1441CD0(a3, *(_QWORD *)(j + 80)) )
        v9 += sub_18A4250(a1, j + 64, a3);
    }
  }
  return v9;
}
