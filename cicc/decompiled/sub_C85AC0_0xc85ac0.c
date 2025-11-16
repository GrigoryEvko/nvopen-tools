// Function: sub_C85AC0
// Address: 0xc85ac0
//
unsigned __int64 __fastcall sub_C85AC0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, char a5)
{
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rbx
  int fd[3]; // [rsp+1Ch] [rbp-14h] BYREF

  result = sub_C85AA0(a1, a2, a3, fd, a4, a5);
  v6 = result;
  if ( !(_DWORD)result )
  {
    close(fd[0]);
    return v6 & 0xFFFFFFFF00000000LL;
  }
  return result;
}
