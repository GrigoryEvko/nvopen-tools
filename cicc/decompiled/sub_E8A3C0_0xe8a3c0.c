// Function: sub_E8A3C0
// Address: 0xe8a3c0
//
__int64 __fastcall sub_E8A3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 result; // rax

  v9 = *(_QWORD *)(a1 + 296);
  v10 = *(_QWORD *)(v9 + 24);
  result = sub_E8EE00(v10, v9, a2, a3);
  if ( a5 )
    return sub_2241130(v10 + 24, 0, *(_QWORD *)(v10 + 32), a4, a5);
  return result;
}
