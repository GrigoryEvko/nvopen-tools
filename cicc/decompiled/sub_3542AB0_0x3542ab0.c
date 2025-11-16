// Function: sub_3542AB0
// Address: 0x3542ab0
//
__int64 __fastcall sub_3542AB0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 result; // rax

  result = (unsigned int)qword_503EC08;
  if ( (int)qword_503EC08 > 0 || (result = *(unsigned int *)(a1 + 3512), (_DWORD)result) )
  {
    *(_DWORD *)(a1 + 3472) = result;
  }
  else
  {
    if ( a2 < a3 )
      a2 = a3;
    *(_DWORD *)(a1 + 3472) = a2;
  }
  return result;
}
