// Function: sub_325F4E0
// Address: 0x325f4e0
//
__int64 __fastcall sub_325F4E0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  if ( a2 > 0x40 )
    return *(_QWORD *)a1;
  result = 0;
  if ( a2 )
    return a1 << (64 - (unsigned __int8)a2) >> (64 - (unsigned __int8)a2);
  return result;
}
