// Function: sub_CA8360
// Address: 0xca8360
//
__int64 __fastcall sub_CA8360(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 232);
  if ( (_DWORD)result )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 224) + 24LL * (unsigned int)result - 8) == a2 )
    {
      result = (unsigned int)(result - 1);
      *(_DWORD *)(a1 + 232) = result;
    }
  }
  return result;
}
