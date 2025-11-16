// Function: sub_3244DB0
// Address: 0x3244db0
//
__int64 *__fastcall sub_3244DB0(__int64 *a1, _QWORD *a2, unsigned __int8 a3)
{
  __int64 *result; // rax
  __int64 v5; // rsi
  __int64 v7; // rsi

  result = (__int64 *)a2[10];
  if ( *((_DWORD *)result + 8) != 3 )
  {
    v5 = a2[7];
    if ( v5 )
    {
      result = (__int64 *)a2[2];
      if ( result )
      {
        if ( (*result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*a1 + 224) + 176LL))(
            *(_QWORD *)(*a1 + 224),
            v5,
            0);
          (*(void (__fastcall **)(_QWORD *, _QWORD))(*a2 + 64LL))(a2, a3);
          result = sub_31F1320(*a1, (__int64)(a2 + 1));
          v7 = a2[25];
          if ( v7 )
            return (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*a1 + 224) + 208LL))(
                                *(_QWORD *)(*a1 + 224),
                                v7,
                                0);
        }
      }
    }
  }
  return result;
}
