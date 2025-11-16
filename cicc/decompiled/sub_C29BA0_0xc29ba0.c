// Function: sub_C29BA0
// Address: 0xc29ba0
//
__int64 __fastcall sub_C29BA0(_QWORD *a1)
{
  __int64 result; // rax

  result = sub_C26F80((__int64)a1);
  if ( !(_DWORD)result )
  {
    result = sub_C299E0(a1);
    if ( !(_DWORD)result )
    {
      sub_C1AFD0();
      return 0;
    }
  }
  return result;
}
