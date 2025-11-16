// Function: sub_D5D740
// Address: 0xd5d740
//
_QWORD *__fastcall sub_D5D740(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  _QWORD *v7; // rdi

  *a1 = a2;
  result = a1 + 9;
  v7 = a1 + 49;
  *(v7 - 48) = a3;
  *(v7 - 47) = a5;
  *(v7 - 46) = a6;
  *((_DWORD *)v7 - 86) = 1;
  *(v7 - 44) = 0;
  *(v7 - 42) = 0;
  *(v7 - 41) = 1;
  do
  {
    if ( result )
      *result = -4096;
    result += 5;
  }
  while ( v7 != result );
  return result;
}
