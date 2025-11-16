// Function: sub_86F7D0
// Address: 0x86f7d0
//
__int64 __fastcall sub_86F7D0(unsigned int a1, _DWORD *a2)
{
  __int64 result; // rax

  result = dword_4F5FD80 | (unsigned int)qword_4F5FD78;
  if ( !(dword_4F5FD80 | (unsigned int)qword_4F5FD78) )
  {
    result = sub_684B30(a1, a2);
    dword_4F5FD80 = 1;
  }
  return result;
}
