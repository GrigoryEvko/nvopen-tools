// Function: sub_3989E20
// Address: 0x3989e20
//
__int64 __fastcall sub_3989E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  _QWORD *v7; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = 0;
  result = sub_39C8280(a2, a3, a4, v8);
  if ( !result )
  {
    v7 = sub_20FAC60(a1 + 64, a5);
    return sub_39CF220(a2, v8[0], v7);
  }
  return result;
}
