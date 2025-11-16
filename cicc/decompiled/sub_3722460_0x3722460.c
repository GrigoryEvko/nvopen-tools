// Function: sub_3722460
// Address: 0x3722460
//
_QWORD *__fastcall sub_3722460(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v4; // r14
  __int64 v5; // rbx
  __int64 v6; // r13
  _QWORD *v7; // r15
  __int64 v8; // r12
  unsigned __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5 >> 1;
        v7 = &v4[v5 >> 1];
        v8 = *a3;
        v11 = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 16LL))(*v7);
        if ( v11 >= (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8) )
          break;
        v4 = v7 + 1;
        v5 = v5 - v6 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v4;
}
