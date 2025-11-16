// Function: sub_3060820
// Address: 0x3060820
//
_QWORD *__fastcall sub_3060820(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  _QWORD *result; // rax
  __int64 v6; // r12
  unsigned __int64 *v7; // rsi
  _QWORD *v8; // [rsp+8h] [rbp-18h] BYREF

  result = (_QWORD *)sub_B82360(*(_QWORD *)(a2 + 8), (__int64)&unk_5040919);
  if ( result )
  {
    result = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, void *))(*result + 104LL))(result, &unk_5040919);
    if ( result )
    {
      v6 = result[22];
      result = (_QWORD *)sub_22077B0(0x10u);
      if ( result )
      {
        result[1] = v6;
        *result = &unk_4A30E28;
      }
      v8 = result;
      v7 = (unsigned __int64 *)a4[2];
      if ( v7 == (unsigned __int64 *)a4[3] )
      {
        return (_QWORD *)sub_3060640(a4 + 1, v7, &v8);
      }
      else
      {
        if ( v7 )
        {
          *v7 = (unsigned __int64)result;
          v7 = (unsigned __int64 *)a4[2];
        }
        a4[2] = (unsigned __int64)(v7 + 1);
      }
    }
  }
  return result;
}
