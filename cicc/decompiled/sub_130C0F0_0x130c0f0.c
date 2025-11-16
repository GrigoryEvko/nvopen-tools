// Function: sub_130C0F0
// Address: 0x130c0f0
//
__int64 __fastcall sub_130C0F0(
        int a1,
        __int64 a2,
        int a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        unsigned __int8 a7)
{
  int v8; // r13d
  __int64 result; // rax
  __int64 v10; // rax
  int v11; // edx

  v8 = a6;
  result = sub_1345C00(a1, a2, a3, (int)a2 + 56, 0, a4, a5, a6, a7);
  if ( !result )
  {
    v10 = sub_130C0D0(a2, 2);
    v11 = a3;
    if ( !v10 || (result = sub_1345C00(a1, a2, a3, (int)a2 + 19496, 0, a4, a5, v8, a7), v11 = a3, !result) )
    {
      result = sub_1345C30(a1, a2, v11, (int)a2 + 38936, 0, a4, a5, v8, a7);
      if ( result )
        _InterlockedAdd64((volatile signed __int64 *)(*(_QWORD *)(a2 + 62224) + 56LL), a4);
    }
  }
  return result;
}
