// Function: sub_AA4AF0
// Address: 0xaa4af0
//
__int64 __fastcall sub_AA4AF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // rsi
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 72);
  v3 = *(_QWORD *)(a2 + 32);
  if ( v3 != *(_QWORD *)(a1 + 32) && a1 + 24 != v3 )
    return sub_B2C300(v2, v3, *(_QWORD *)(a1 + 72));
  return result;
}
