// Function: sub_22529E0
// Address: 0x22529e0
//
__int64 __fastcall sub_22529E0(__int64 a1, __int64 a2, _QWORD **a3)
{
  __int64 result; // rax
  _QWORD *v5; // [rsp+8h] [rbp-20h] BYREF

  v5 = *a3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2) )
    v5 = (_QWORD *)*v5;
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD **, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2, &v5, 1);
  if ( (_BYTE)result )
    *a3 = v5;
  return result;
}
