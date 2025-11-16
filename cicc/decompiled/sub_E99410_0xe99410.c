// Function: sub_E99410
// Address: 0xe99410
//
__int64 __fastcall sub_E99410(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
  {
    *(_QWORD *)(result + 24) = a2;
    *(_DWORD *)(result + 64) = a3;
  }
  return result;
}
