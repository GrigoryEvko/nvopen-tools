// Function: sub_F03EF0
// Address: 0xf03ef0
//
__int64 __fastcall sub_F03EF0(unsigned __int64 a1)
{
  __int64 result; // rax

  LODWORD(result) = 0;
  do
  {
    result = (unsigned int)(result + 1);
    a1 >>= 7;
  }
  while ( a1 );
  return result;
}
