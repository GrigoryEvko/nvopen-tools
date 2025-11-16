// Function: sub_16A7890
// Address: 0x16a7890
//
__int64 __fastcall sub_16A7890(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned int a6,
        char a7)
{
  unsigned int v9; // ebx
  __int64 v10; // r13
  unsigned __int64 v11; // r14
  __int64 v12; // rsi
  bool v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  bool v19; // cf
  __int64 v20; // r8
  __int64 result; // rax

  v9 = a5;
  if ( a5 > a6 )
    v9 = a6;
  if ( v9 )
  {
    v10 = (unsigned int)a3;
    v11 = HIDWORD(a3);
    v12 = 0;
    v13 = a3 == 0;
    do
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(a2 + 8 * v12);
        if ( !v15 || v13 )
        {
          v20 = 0;
        }
        else
        {
          v16 = (unsigned int)v15;
          v17 = HIDWORD(v15);
          v18 = ((v10 * v17) << 32) + v10 * v16 + ((v11 * v16) << 32);
          v19 = __CFADD__(v18, a4);
          a4 += v18;
          v20 = ((__PAIR128__((v10 * v17) >> 32, (v10 * v17) << 32)
                + __PAIR128__((v11 * v16) >> 32, v10 * v16)
                + __PAIR128__(v11 * v17, (v11 * v16) << 32)) >> 64)
              - (!v19
               - 1LL);
        }
        if ( a7 )
          break;
        *(_QWORD *)(a1 + 8 * v12++) = a4;
        a4 = v20;
        if ( v9 == v12 )
          goto LABEL_13;
      }
      v14 = *(_QWORD *)(a1 + 8 * v12) + a4;
      if ( __CFADD__(*(_QWORD *)(a1 + 8 * v12), a4) )
        a4 = v20 + 1;
      else
        a4 = v20;
      *(_QWORD *)(a1 + 8 * v12++) = v14;
    }
    while ( v9 != v12 );
  }
LABEL_13:
  if ( a5 < a6 )
  {
    *(_QWORD *)(a1 + 8LL * a5) = a4;
    return 0;
  }
  else
  {
    result = 1;
    if ( !a4 )
    {
      result = 0;
      if ( a3 )
      {
        if ( a5 != a6 )
        {
          do
          {
            if ( *(_QWORD *)(a2 + 8LL * a6) )
              return 1;
            ++a6;
          }
          while ( a5 > a6 );
          return 0;
        }
      }
    }
  }
  return result;
}
