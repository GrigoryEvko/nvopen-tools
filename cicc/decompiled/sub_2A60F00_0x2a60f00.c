// Function: sub_2A60F00
// Address: 0x2a60f00
//
__int64 __fastcall sub_2A60F00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 *v8; // rcx
  __int64 v9; // r8
  __int64 i; // r12
  __int64 j; // r15
  int v13; // ecx
  int v14; // r10d
  unsigned int v15; // [rsp+14h] [rbp-3Ch]

  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 56LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 56 * v5) )
      {
        v15 = *((_DWORD *)v8 + 12);
        goto LABEL_5;
      }
    }
    else
    {
      v13 = 1;
      while ( v9 != -4096 )
      {
        v14 = v13 + 1;
        v7 = (v5 - 1) & (v13 + v7);
        v8 = (__int64 *)(v6 + 56LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v13 = v14;
      }
    }
  }
  v15 = 0;
LABEL_5:
  for ( i = *(_QWORD *)(a2 + 144); a2 + 128 != i; i = sub_220EF30(i) )
  {
    for ( j = *(_QWORD *)(i + 64); i + 48 != j; j = sub_220EF30(j) )
    {
      if ( sub_2A60EC0(j + 48, a3, *(_BYTE *)(a1 + 40)) )
        v15 += sub_2A60F00(a1, j + 48, a3);
    }
  }
  return v15;
}
