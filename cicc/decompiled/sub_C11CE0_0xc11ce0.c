// Function: sub_C11CE0
// Address: 0xc11ce0
//
__int64 __fastcall sub_C11CE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD *v3; // rdx
  _QWORD *v4; // r12
  _QWORD *v5; // r15
  __int64 *v6; // rbx
  __int64 *i; // r13
  __int64 *v8; // rsi
  __int64 v9; // rdx
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8

  result = sub_C14B60(a2);
  if ( (_QWORD *)result != v3 )
  {
    v4 = (_QWORD *)result;
    v5 = v3;
    do
    {
      v6 = (__int64 *)v4[1];
      for ( i = (__int64 *)v4[2];
            i != v6;
            result = (**(__int64 (__fastcall ***)(_QWORD, _QWORD *, __int64, __int64, __int64))a1)(
                       *(_QWORD *)(*(_QWORD *)a1 + 8LL),
                       v10,
                       v9,
                       v11,
                       v12) )
      {
        if ( (*(_BYTE *)(*v4 + 8LL) & 1) != 0 )
        {
          v8 = *(__int64 **)(*v4 - 8LL);
          v9 = *v8;
          v10 = v8 + 3;
        }
        else
        {
          v9 = 0;
          v10 = 0;
        }
        v11 = *v6;
        v12 = v6[1];
        v6 += 2;
      }
      v4 += 4;
    }
    while ( v5 != v4 );
  }
  return result;
}
