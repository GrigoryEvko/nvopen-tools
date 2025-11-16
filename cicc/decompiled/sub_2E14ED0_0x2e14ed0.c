// Function: sub_2E14ED0
// Address: 0x2e14ed0
//
__int64 __fastcall sub_2E14ED0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rdi
  __int16 *v6; // r15
  unsigned int v7; // r13d
  __int64 result; // rax
  __int64 v9; // r14
  __int64 *v10; // rsi
  __int64 v11; // rsi
  int v12; // esi
  unsigned int v13; // [rsp+0h] [rbp-40h]
  unsigned __int64 v14; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 16);
  v6 = (__int16 *)(*(_QWORD *)(v5 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * a2 + 16) >> 12));
  v7 = *(_DWORD *)(*(_QWORD *)(v5 + 8) + 24LL * a2 + 16) & 0xFFF;
  v14 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  result = (a3 >> 1) & 3;
  v13 = result;
  do
  {
    if ( !v6 )
      break;
    result = v7;
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * v7);
    if ( v9 )
    {
      v10 = (__int64 *)sub_2E09D00((__int64 *)v9, a3);
      result = *(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8);
      if ( v10 != (__int64 *)result )
      {
        result = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3;
        if ( (unsigned int)result <= (*(_DWORD *)(v14 + 24) | v13) )
        {
          v11 = v10[2];
          if ( v11 )
            result = sub_2E0A600(v9, v11);
        }
      }
    }
    v12 = *v6++;
    v7 += v12;
  }
  while ( (_WORD)v12 );
  return result;
}
