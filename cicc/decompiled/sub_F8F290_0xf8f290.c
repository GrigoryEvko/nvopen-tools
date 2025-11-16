// Function: sub_F8F290
// Address: 0xf8f290
//
__int64 __fastcall sub_F8F290(__int64 a1, int a2)
{
  __int64 v2; // rax

  v2 = 32;
  if ( a2 != -2 )
    v2 = 32LL * (unsigned int)(2 * a2 + 3);
  return *(_QWORD *)(*(_QWORD *)(a1 - 8) + v2);
}
