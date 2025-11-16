// Function: sub_E993E0
// Address: 0xe993e0
//
__int64 __fastcall sub_E993E0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
  {
    *(_QWORD *)(result + 16) = a2;
    *(_DWORD *)(result + 60) = a3;
  }
  return result;
}
