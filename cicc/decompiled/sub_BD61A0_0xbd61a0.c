// Function: sub_BD61A0
// Address: 0xbd61a0
//
__int64 __fastcall sub_BD61A0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    if ( result != -4096 && result != -8192 )
      result = sub_BD60C0((_QWORD *)(a1 + 8));
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
