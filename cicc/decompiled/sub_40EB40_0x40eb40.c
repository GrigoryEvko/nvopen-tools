// Function: sub_40EB40
// Address: 0x40eb40
//
__int64 __fastcall sub_40EB40(unsigned int *a1, __int64 a2, int a3, int a4, int a5, int a6, char a7)
{
  __int64 result; // rax

  result = *a1;
  if ( (unsigned int)result <= 1 )
    return sub_40E56D(a1, a2, a3, a4, a5, a6, a7);
  if ( (_DWORD)result == 2 )
  {
    --a1[6];
    *((_BYTE *)a1 + 28) = 1;
  }
  return result;
}
