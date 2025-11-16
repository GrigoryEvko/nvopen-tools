// Function: sub_CCC540
// Address: 0xccc540
//
__int64 __fastcall sub_CCC540(__int64 a1, unsigned int *a2)
{
  unsigned int v3; // r13d
  unsigned int i; // ebx
  unsigned int *v5; // rsi
  _QWORD v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v3 = 2;
  }
  else if ( !v3 )
  {
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
  }
  for ( i = 0; i < v3; ++i )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD *))(*(_QWORD *)a1 + 72LL))(a1, i, v7) )
    {
      v5 = a2 + 1;
      if ( !i )
        v5 = a2;
      sub_CCC2C0(a1, v5);
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, v7[0]);
    }
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
}
