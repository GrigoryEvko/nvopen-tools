// Function: sub_2BF04B0
// Address: 0x2bf04b0
//
__int64 __fastcall sub_2BF04B0(__int64 a1)
{
  __int64 result; // rax

  for ( result = a1; !*(_BYTE *)(result + 8); result = *(_QWORD *)(result + 112) )
    ;
  return result;
}
