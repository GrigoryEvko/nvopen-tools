// Function: sub_2BF0500
// Address: 0x2bf0500
//
__int64 __fastcall sub_2BF0500(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; !*(_BYTE *)(result + 8); result = *(_QWORD *)(result + 120) )
    ;
  return result;
}
