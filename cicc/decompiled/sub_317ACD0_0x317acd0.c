// Function: sub_317ACD0
// Address: 0x317acd0
//
unsigned __int64 __fastcall sub_317ACD0(__int64 a1)
{
  unsigned __int64 v2; // r12
  unsigned int v3; // eax
  __int64 *v4; // rdi
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 *v12; // rsi
  __int64 v13; // rdi
  signed __int64 v14; // rax
  bool v15; // of
  int v17; // esi

  v2 = 0;
  while ( 1 )
  {
    v3 = *(_DWORD *)(a1 + 168);
    if ( !v3 )
      return v2;
    v4 = *(__int64 **)(a1 + 56);
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8LL * v3 - 8);
    *(_DWORD *)(a1 + 168) = v3 - 1;
    v6 = *(_QWORD *)(v5 + 40);
    if ( (unsigned __int8)sub_2A64220(v4, v6) )
    {
      v9 = *(unsigned int *)(a1 + 120);
      v10 = *(_QWORD *)(a1 + 104);
      if ( !(_DWORD)v9 )
        goto LABEL_7;
      v7 = (unsigned int)(v9 - 1);
      v11 = v7 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v12 = (__int64 *)(v10 + 8LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
      {
LABEL_6:
        if ( v12 == (__int64 *)(v10 + 8 * v9) )
          goto LABEL_7;
      }
      else
      {
        v17 = 1;
        while ( v13 != -4096 )
        {
          v8 = (unsigned int)(v17 + 1);
          v11 = v7 & (v17 + v11);
          v12 = (__int64 *)(v10 + 8LL * v11);
          v13 = *v12;
          if ( v6 == *v12 )
            goto LABEL_6;
          v17 = v8;
        }
LABEL_7:
        v14 = sub_317A680(a1, v5, 0, 0, v7, v8);
        v15 = __OFADD__(v14, v2);
        v2 += v14;
        if ( v15 )
        {
          v2 = 0x8000000000000000LL;
          if ( v14 > 0 )
            v2 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
    }
  }
}
