// Function: sub_168C3C0
// Address: 0x168c3c0
//
_QWORD *__fastcall sub_168C3C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *result; // rax
  _QWORD *v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx

  sub_168ED70(a2);
  v2 = sub_168E720(a2);
  result = (_QWORD *)sub_168E770(a2);
  if ( result != (_QWORD *)v2 )
  {
    v4 = result;
    do
    {
      while ( 1 )
      {
        v5 = 2048;
        v6 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v2 + 8LL) - 1);
        if ( (unsigned int)v6 <= 5 )
          v5 = dword_42AE480[v6];
        (**(void (__fastcall ***)(_QWORD, __int64, _QWORD, __int64))a1)(
          *(_QWORD *)(*(_QWORD *)a1 + 8LL),
          *(_QWORD *)v2 + 16LL,
          **(_QWORD **)v2,
          v5);
        result = *(_QWORD **)(v2 + 8);
        v7 = v2 + 8;
        if ( result == (_QWORD *)-8LL || !result )
          break;
        v2 += 8;
        if ( v4 == (_QWORD *)v7 )
          return result;
      }
      result = (_QWORD *)(v2 + 16);
      do
      {
        do
        {
          v8 = *result;
          v2 = (__int64)result++;
        }
        while ( !v8 );
      }
      while ( v8 == -8 );
    }
    while ( v4 != (_QWORD *)v2 );
  }
  return result;
}
