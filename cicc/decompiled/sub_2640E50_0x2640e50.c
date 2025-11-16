// Function: sub_2640E50
// Address: 0x2640e50
//
_QWORD *__fastcall sub_2640E50(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  _QWORD *result; // rax
  __int64 v6; // rsi
  _QWORD *v7; // rdx
  _QWORD *v8; // rsi
  __int64 v9; // rcx

  v4 = a2[1] - *a2;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF0LL )
      sub_4261EA(a1, a2, a3);
    result = (_QWORD *)sub_22077B0(v4);
  }
  else
  {
    result = 0;
  }
  *a1 = result;
  a1[1] = result;
  a1[2] = (char *)result + v4;
  v6 = a2[1];
  v7 = (_QWORD *)*a2;
  if ( *a2 == v6 )
  {
    a1[1] = result;
  }
  else
  {
    v8 = (_QWORD *)((char *)result + v6 - (_QWORD)v7);
    do
    {
      if ( result )
      {
        *result = *v7;
        v9 = v7[1];
        result[1] = v9;
        if ( v9 )
        {
          if ( &_pthread_key_create )
            _InterlockedAdd((volatile signed __int32 *)(v9 + 8), 1u);
          else
            ++*(_DWORD *)(v9 + 8);
        }
      }
      result += 2;
      v7 += 2;
    }
    while ( result != v8 );
    a1[1] = v8;
  }
  return result;
}
