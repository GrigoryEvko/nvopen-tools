// Function: sub_16C64C0
// Address: 0x16c64c0
//
unsigned __int64 __fastcall sub_16C64C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rbx
  int fd[3]; // [rsp+1Ch] [rbp-14h] BYREF

  result = sub_16C64B0(a1, a2, a3, fd, a4);
  v5 = result;
  if ( !(_DWORD)result )
  {
    close(fd[0]);
    return v5 & 0xFFFFFFFF00000000LL;
  }
  return result;
}
