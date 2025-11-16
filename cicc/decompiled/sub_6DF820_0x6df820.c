// Function: sub_6DF820
// Address: 0x6df820
//
__int64 __fastcall sub_6DF820(_QWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = sub_8D2930(*a1);
  if ( (_DWORD)result )
  {
    result = sub_6DEAC0((__int64)a1);
    if ( (_DWORD)result )
    {
      a2[20] = 1;
      a2[18] = 1;
    }
  }
  else
  {
    a2[19] = 1;
  }
  return result;
}
