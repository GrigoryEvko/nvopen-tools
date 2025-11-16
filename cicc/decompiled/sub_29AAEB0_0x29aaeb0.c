// Function: sub_29AAEB0
// Address: 0x29aaeb0
//
__int64 __fastcall sub_29AAEB0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 *v5; // r13
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 i; // r12
  int v9; // edx
  int v10; // r9d
  unsigned int v11; // eax
  __int64 v12; // r8
  __int64 v13; // rsi
  __int64 v14; // rdi
  int v15; // eax

  v5 = sub_BD4070(a3, a2);
  v6 = *(_QWORD *)(**(_QWORD **)(a1 + 88) + 72LL);
  v7 = *(_QWORD *)(v6 + 80);
  for ( i = v6 + 72; i != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    v13 = v7 - 24;
    v14 = *(_QWORD *)(a1 + 64);
    if ( !v7 )
      v13 = 0;
    v15 = *(_DWORD *)(a1 + 80);
    if ( v15 )
    {
      v9 = v15 - 1;
      v10 = 1;
      v11 = (v15 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v12 = *(_QWORD *)(v14 + 8LL * v11);
      if ( v13 != v12 )
      {
        while ( v12 != -4096 )
        {
          v11 = v9 & (v10 + v11);
          v12 = *(_QWORD *)(v14 + 8LL * v11);
          if ( v13 == v12 )
            goto LABEL_4;
          ++v10;
        }
        if ( (unsigned __int8)sub_29AAD40(a2, v13, (__int64)v5) )
          return 0;
      }
    }
    else if ( (unsigned __int8)sub_29AAD40(a2, v13, (__int64)v5) )
    {
      return 0;
    }
LABEL_4:
    ;
  }
  return 1;
}
