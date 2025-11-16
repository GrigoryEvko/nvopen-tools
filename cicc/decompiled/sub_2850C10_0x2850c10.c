// Function: sub_2850C10
// Address: 0x2850c10
//
__int64 __fastcall sub_2850C10(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  if ( a2 > 0x40 )
    return *(_QWORD *)a1;
  result = 0;
  if ( a2 )
    return a1 << (64 - (unsigned __int8)a2) >> (64 - (unsigned __int8)a2);
  return result;
}
