// Function: sub_B33C40
// Address: 0xb33c40
//
__int64 __fastcall sub_B33C40(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 result; // rax
  _QWORD v13[8]; // [rsp+20h] [rbp-40h] BYREF

  v7 = sub_AA4B30(*(_QWORD *)(a1 + 48));
  v13[0] = *(_QWORD *)(a3 + 8);
  v8 = sub_B6E160(v7, a2, v13, 1);
  result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 80) + 128LL))(
             *(_QWORD *)(a1 + 80),
             a2,
             a3,
             a4,
             **(_QWORD **)(*(_QWORD *)(v8 + 24) + 16LL),
             0);
  if ( !result )
  {
    v13[0] = a3;
    v13[1] = a4;
    return sub_B33A00(a1, v8, (int)v13, 2, a6, a5, 0, 0);
  }
  return result;
}
