// Function: sub_F53AB0
// Address: 0xf53ab0
//
__int64 __fastcall sub_F53AB0(int a1)
{
  __int64 v1; // rdi
  __int64 result; // rax

  v1 = (unsigned int)(a1 - 13);
  result = 0;
  if ( (unsigned int)v1 <= 0x11 )
    return *(_QWORD *)&asc_3F8A8A0[8 * v1];
  return result;
}
