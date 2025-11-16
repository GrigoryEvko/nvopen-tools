// Function: sub_1697F10
// Address: 0x1697f10
//
__int64 __fastcall sub_1697F10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rdi
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 *v12; // rdx
  __int64 result; // rax

  sub_16977B0(a1, a2, a3, a4, a5, a6);
  v8 = *(_QWORD **)(a1 + 112);
  v9 = *(_QWORD **)(a1 + 104);
  v10 = (__int64)(*(_QWORD *)(a1 + 112) - (_QWORD)v9) >> 4;
  if ( (__int64)(*(_QWORD *)(a1 + 112) - (_QWORD)v9) > 0 )
  {
    do
    {
      while ( 1 )
      {
        v11 = v10 >> 1;
        v12 = &v9[2 * (v10 >> 1)];
        if ( *v12 >= a2 )
          break;
        v9 = v12 + 2;
        v10 = v10 - v11 - 1;
        if ( v10 <= 0 )
          goto LABEL_6;
      }
      v10 >>= 1;
    }
    while ( v11 > 0 );
  }
LABEL_6:
  result = 0;
  if ( v8 != v9 && *v9 == a2 )
    return v9[1];
  return result;
}
