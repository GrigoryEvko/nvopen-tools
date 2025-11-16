// Function: sub_FFE120
// Address: 0xffe120
//
__int64 __fastcall sub_FFE120(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r8d

  v4 = 0;
  if ( 100 * a2 >= a4 * (unsigned __int64)(unsigned int)qword_4F8ED28 )
    LOBYTE(v4) = 100 * a2 >= (unsigned __int64)(unsigned int)qword_4F8EC48 * a3;
  return v4;
}
