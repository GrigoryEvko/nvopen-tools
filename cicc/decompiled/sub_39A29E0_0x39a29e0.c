// Function: sub_39A29E0
// Address: 0x39a29e0
//
unsigned __int64 __fastcall sub_39A29E0(_QWORD *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 v7; // rdi
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rdx

  result = 2LL - *(unsigned int *)(a3 + 8);
  v7 = *(_QWORD *)(a3 + 8 * result);
  if ( v7 )
  {
    result = sub_161E970(v7);
    if ( v10 )
    {
      result = *(unsigned int *)(a3 + 28);
      if ( (result & 4) == 0 )
      {
        v11 = *(unsigned int *)(a3 + 8);
        v12 = a1[25];
        v13 = *(_QWORD *)(a3 + 8 * (2 - v11));
        if ( v13 )
          v13 = sub_161E970(*(_QWORD *)(a3 + 8 * (2 - v11)));
        else
          v14 = 0;
        sub_3990480(v12, v13, v14, a4);
        if ( !a2 )
          return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, unsigned __int8 *))(*a1 + 16LL))(a1, a3, a4, a2);
        result = *a2;
        if ( (unsigned __int8)result <= 0x1Fu )
        {
          v15 = 2148630528LL;
          if ( _bittest64(&v15, result) )
            return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, unsigned __int8 *))(*a1 + 16LL))(
                     a1,
                     a3,
                     a4,
                     a2);
        }
      }
    }
  }
  return result;
}
