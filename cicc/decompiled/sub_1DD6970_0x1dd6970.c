// Function: sub_1DD6970
// Address: 0x1dd6970
//
bool __fastcall sub_1DD6970(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rsi
  __int64 v4; // [rsp+8h] [rbp-8h] BYREF

  v4 = a2;
  v2 = *(_QWORD **)(a1 + 96);
  return v2 != sub_1DD4F70(*(_QWORD **)(a1 + 88), (__int64)v2, &v4);
}
