// Function: sub_D135E0
// Address: 0xd135e0
//
__int64 __fastcall sub_D135E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 32);
  *(_BYTE *)(a1 + 25) = 1;
  v2 = *(_QWORD *)(v1 + 80);
  if ( !v2 )
    BUG();
  result = *(_QWORD *)(v2 + 32);
  if ( result )
    result -= 24;
  *(_QWORD *)(a1 + 8) = result;
  return result;
}
