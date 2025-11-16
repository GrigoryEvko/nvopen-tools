// Function: sub_3880F40
// Address: 0x3880f40
//
__int64 __fastcall sub_3880F40(unsigned __int8 **a1)
{
  unsigned __int8 *v1; // rdx
  __int64 result; // rax

  v1 = (*a1)++;
  result = *v1;
  if ( !(_BYTE)result )
  {
    if ( v1 == &a1[2][(_QWORD)a1[1]] )
    {
      *a1 = v1;
      return 0xFFFFFFFFLL;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
