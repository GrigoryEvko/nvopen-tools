// Function: sub_AA4AC0
// Address: 0xaa4ac0
//
__int64 __fastcall sub_AA4AC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( *(_QWORD *)(a1 + 32) != a2 && a1 + 24 != a2 )
    return sub_B2C300(*(_QWORD *)(a1 + 72), a2, *(_QWORD *)(a1 + 72));
  return result;
}
