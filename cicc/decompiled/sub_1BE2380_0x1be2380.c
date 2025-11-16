// Function: sub_1BE2380
// Address: 0x1be2380
//
__int64 __fastcall sub_1BE2380(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; *(_BYTE *)(result + 8) == 1; result = *(_QWORD *)(result + 120) )
    ;
  return result;
}
