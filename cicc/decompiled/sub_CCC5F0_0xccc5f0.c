// Function: sub_CCC5F0
// Address: 0xccc5f0
//
__int64 __fastcall sub_CCC5F0(__int64 a1, __int64 a2, unsigned int *a3, unsigned __int8 a4)
{
  __int64 result; // rax
  char v6; // [rsp+7h] [rbp-19h] BYREF
  _QWORD v7[3]; // [rsp+8h] [rbp-18h] BYREF

  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             a2,
             a4,
             0,
             &v6,
             v7);
  if ( (_BYTE)result )
  {
    sub_CCC2C0(a1, a3);
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v7[0]);
  }
  return result;
}
