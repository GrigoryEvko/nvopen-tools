// Function: sub_356D550
// Address: 0x356d550
//
_QWORD *__fastcall sub_356D550(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  _QWORD *result; // rax
  __int64 *v8; // r12
  __int64 *v9; // rax
  __int64 v10; // r15
  __int64 *v11; // rax
  unsigned __int64 *v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 *v14; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 16);
  if ( a2 )
  {
    v6 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    result = (_QWORD *)v6;
  }
  else
  {
    v6 = 0;
    result = 0;
  }
  if ( (unsigned int)result < *(_DWORD *)(v5 + 56) )
  {
    result = *(_QWORD **)(v5 + 48);
    v8 = (__int64 *)result[v6];
    if ( v8 )
    {
      v13 = a2;
      v14 = 0;
      do
      {
        v9 = (__int64 *)sub_35684C0(a1, v8, a3);
        v8 = v9;
        if ( !v9 )
          break;
        v10 = *v9;
        if ( !*v9 )
          break;
        if ( sub_35681D0(a1, a2, *v9) )
        {
          v11 = sub_356C580(a1, a2, v10);
          v12 = (unsigned __int64 *)v14;
          if ( v14 )
          {
            v14 = v11;
            sub_356CAD0(v11, v12, 0);
            v13 = v10;
          }
          else
          {
            v13 = v10;
            v14 = v11;
          }
        }
      }
      while ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 8), a2, v10) );
      result = (_QWORD *)v13;
      if ( v13 != a2 )
        return sub_356BEB0(a1, a2, v13, a3);
    }
  }
  return result;
}
