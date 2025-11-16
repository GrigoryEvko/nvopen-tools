// Function: sub_1F43CC0
// Address: 0x1f43cc0
//
__int64 __fastcall sub_1F43CC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned int a7,
        _BYTE *a8)
{
  __int64 v10; // rax
  __int64 (*v11)(); // r10
  __int64 result; // rax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]

  v13 = a4;
  v14 = a5;
  v10 = sub_1F58E60(&v13, a2);
  if ( (unsigned int)sub_15A9FE0(a3, v10) <= a7 )
  {
    result = 1;
    if ( a8 )
      *a8 = 1;
  }
  else
  {
    v11 = *(__int64 (**)())(*(_QWORD *)a1 + 448LL);
    result = 0;
    if ( v11 != sub_1D12D60 )
      return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, _QWORD, _BYTE *))v11)(
               a1,
               (unsigned int)v13,
               v14,
               a6,
               a7,
               a8);
  }
  return result;
}
