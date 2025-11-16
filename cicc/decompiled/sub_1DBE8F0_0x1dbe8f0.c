// Function: sub_1DBE8F0
// Address: 0x1dbe8f0
//
unsigned __int64 __fastcall sub_1DBE8F0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v6; // edi
  _WORD *v7; // r8
  __int16 v8; // dx
  unsigned __int16 *v9; // rdi
  unsigned __int16 v10; // r14
  unsigned __int16 *v11; // rbx
  unsigned __int64 result; // rax
  __int64 v13; // r15
  __int64 *v14; // rsi
  __int64 v15; // rsi

  v4 = *(_QWORD *)(a1 + 248);
  if ( !v4 )
    BUG();
  v6 = *(_DWORD *)(*(_QWORD *)(v4 + 8) + 24LL * a2 + 16);
  v7 = (_WORD *)(*(_QWORD *)(v4 + 56) + 2LL * (v6 >> 4));
  v8 = a2 * (v6 & 0xF);
  v9 = v7 + 1;
  v10 = *v7 + v8;
  while ( 1 )
  {
    v11 = v9;
    result = a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v9 )
      break;
    while ( 1 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v10);
      if ( v13 )
      {
        v14 = (__int64 *)sub_1DB3C70((__int64 *)v13, a3);
        if ( v14 != (__int64 *)(*(_QWORD *)v13 + 24LL * *(unsigned int *)(v13 + 8))
          && (*(_DWORD *)((*v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v14 >> 1) & 3) <= (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(a3 >> 1)
                                                                                                 & 3) )
        {
          v15 = v14[2];
          if ( v15 )
            sub_1DB4670(v13, v15);
        }
      }
      result = *v11;
      v9 = 0;
      ++v11;
      if ( !(_WORD)result )
        break;
      v10 += result;
      if ( !v11 )
        return result;
    }
  }
  return result;
}
