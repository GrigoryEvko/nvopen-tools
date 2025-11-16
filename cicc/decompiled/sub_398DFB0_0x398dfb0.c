// Function: sub_398DFB0
// Address: 0x398dfb0
//
_QWORD *__fastcall sub_398DFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *result; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = 0;
  result = (_QWORD *)sub_39C8280(a2, a3, a4, &v7);
  if ( !result )
  {
    v8[0] = a5;
    result = sub_20FABF0((_QWORD *)(a1 + 184), v8);
    if ( result )
      return (_QWORD *)sub_39CF220(a2, v7, result + 2);
  }
  return result;
}
