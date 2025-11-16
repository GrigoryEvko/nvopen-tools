// Function: sub_2E88D70
// Address: 0x2e88d70
//
__int64 __fastcall sub_2E88D70(__int64 a1, unsigned __int16 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  if ( *(_QWORD *)(a1 + 24) )
  {
    v2 = sub_2E88D60(a1);
    sub_2E78D60(v2);
  }
  *(_QWORD *)(a1 + 16) = a2;
  result = *a2;
  *(_WORD *)(a1 + 68) = result;
  return result;
}
