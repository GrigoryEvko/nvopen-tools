// Function: sub_13468F0
// Address: 0x13468f0
//
unsigned __int64 __fastcall sub_13468F0(void *a1, size_t a2, __int64 a3, _BYTE *a4, _BYTE *a5)
{
  unsigned __int64 result; // rax

  result = sub_130CA40(a1, a2, a3, a5);
  if ( result )
  {
    if ( *a5 )
      *a4 = 1;
  }
  return result;
}
