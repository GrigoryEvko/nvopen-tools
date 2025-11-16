// Function: sub_263DA70
// Address: 0x263da70
//
__int64 __fastcall sub_263DA70(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == result )
    return 0;
  return result;
}
