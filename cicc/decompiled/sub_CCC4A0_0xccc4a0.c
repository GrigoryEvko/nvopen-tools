// Function: sub_CCC4A0
// Address: 0xccc4a0
//
__int64 __fastcall sub_CCC4A0(__int64 a1, __int64 a2, unsigned int *a3, _DWORD *a4, unsigned __int8 a5)
{
  char v8; // al
  _BOOL8 v9; // rcx
  __int64 result; // rax
  char v11; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *a3 == *a4;
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             a2,
             a5,
             v9,
             &v11,
             v12);
  if ( (_BYTE)result )
  {
    sub_CCC2C0(a1, a3);
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v12[0]);
  }
  else if ( v11 )
  {
    result = (unsigned int)*a4;
    *a3 = result;
  }
  return result;
}
