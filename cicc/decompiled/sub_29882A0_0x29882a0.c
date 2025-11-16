// Function: sub_29882A0
// Address: 0x29882a0
//
__int64 __fastcall sub_29882A0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 80) == result )
    return 0;
  return result;
}
