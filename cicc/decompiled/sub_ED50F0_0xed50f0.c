// Function: sub_ED50F0
// Address: 0xed50f0
//
__int64 __fastcall sub_ED50F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 *v12; // rdx
  __int64 result; // rax

  if ( !*(_BYTE *)(a1 + 392) )
    sub_ED4D20(a1, a2, a3, a4, a5, a6);
  v8 = *(_QWORD **)(a1 + 160);
  v9 = *(_QWORD **)(a1 + 152);
  v10 = (__int64)(*(_QWORD *)(a1 + 160) - (_QWORD)v9) >> 4;
  if ( (__int64)(*(_QWORD *)(a1 + 160) - (_QWORD)v9) > 0 )
  {
    do
    {
      while ( 1 )
      {
        v11 = v10 >> 1;
        v12 = &v9[2 * (v10 >> 1)];
        if ( a2 <= *v12 )
          break;
        v9 = v12 + 2;
        v10 = v10 - v11 - 1;
        if ( v10 <= 0 )
          goto LABEL_8;
      }
      v10 >>= 1;
    }
    while ( v11 > 0 );
  }
LABEL_8:
  result = 0;
  if ( v8 != v9 && *v9 == a2 )
    return v9[1];
  return result;
}
