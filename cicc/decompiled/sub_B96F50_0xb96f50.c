// Function: sub_B96F50
// Address: 0xb96f50
//
__int64 __fastcall sub_B96F50(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 8 * a2);
  if ( v2 )
    return sub_B96E90(a1 + 8 * a2, v2, a1 & 0xFFFFFFFFFFFFFFFCLL | 2);
  return result;
}
