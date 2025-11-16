// Function: sub_E99460
// Address: 0xe99460
//
__int64 __fastcall sub_E99460(__int64 a1, int a2)
{
  __int64 result; // rax

  result = sub_E99320(a1);
  if ( result )
    *(_DWORD *)(result + 84) = a2;
  return result;
}
