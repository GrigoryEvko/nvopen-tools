// Function: sub_D00120
// Address: 0xd00120
//
unsigned __int64 __fastcall sub_D00120(unsigned __int8 *a1, __int64 *a2, __int64 a3, char a4)
{
  unsigned __int64 result; // rax
  char v6; // [rsp+1Eh] [rbp-22h] BYREF
  char v7; // [rsp+1Fh] [rbp-21h] BYREF

  result = sub_BD4FF0(a1, a3, &v6, &v7);
  if ( v6 && a4 )
    result = 0;
  if ( *a2 >= 0 && result < (*a2 & 0x3FFFFFFFFFFFFFFFuLL) )
    return *a2 & 0x3FFFFFFFFFFFFFFFLL;
  return result;
}
