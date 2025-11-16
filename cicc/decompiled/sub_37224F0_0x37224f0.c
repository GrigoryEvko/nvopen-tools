// Function: sub_37224F0
// Address: 0x37224f0
//
_QWORD *__fastcall sub_37224F0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v6; // r13
  __int64 *v7; // r15
  __int64 v8; // r14
  _QWORD *v10; // [rsp+0h] [rbp-40h]
  unsigned __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 3;
  v10 = a1;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v4 >> 1;
        v7 = &v10[v4 >> 1];
        v8 = *v7;
        v11 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a3 + 16LL))(*a3);
        if ( v11 < (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8) )
          break;
        v4 = v4 - v6 - 1;
        v10 = v7 + 1;
        if ( v4 <= 0 )
          return v10;
      }
      v4 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v10;
}
