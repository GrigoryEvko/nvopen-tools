// Function: sub_1DD5D10
// Address: 0x1dd5d10
//
__int64 __fastcall sub_1DD5D10(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rdi

  result = *(_QWORD *)(a1 + 32);
  for ( i = a1 + 24; i != result; result = *(_QWORD *)(result + 8) )
  {
    if ( **(_WORD **)(result + 16) != 45 && **(_WORD **)(result + 16) )
      break;
  }
  return result;
}
