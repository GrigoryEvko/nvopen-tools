// Function: sub_131D390
// Address: 0x131d390
//
__int64 __fastcall sub_131D390(unsigned __int16 *a1, unsigned int a2, __int64 *a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx

  result = a2;
  *a3 = 16;
  if ( (_DWORD)result )
  {
    v7 = (__int64)&a1[(unsigned int)(result - 1) + 1];
    result = 16;
    do
    {
      v8 = *a1++;
      result += 8 * v8;
      *a3 = result;
    }
    while ( (unsigned __int16 *)v7 != a1 );
  }
  *a4 = 4096;
  return result;
}
