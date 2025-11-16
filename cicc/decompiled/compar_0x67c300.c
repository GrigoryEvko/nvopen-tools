// Function: compar
// Address: 0x67c300
//
__int64 __fastcall compar(unsigned __int16 *a1, unsigned __int16 *a2)
{
  __int64 result; // rax

  result = (unsigned int)(*(_DWORD *)a1 - *(_DWORD *)a2);
  if ( *(_DWORD *)a1 - *(_DWORD *)a2 > 0
    || *(_DWORD *)a1 == *(_DWORD *)a2 && (result = a1[2] - (unsigned int)a2[2], (int)result >= 0) )
  {
    qword_4CFDE98 = (__int64)a2;
  }
  return result;
}
