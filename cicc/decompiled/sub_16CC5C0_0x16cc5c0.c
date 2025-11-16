// Function: sub_16CC5C0
// Address: 0x16cc5c0
//
__int64 sub_16CC5C0()
{
  volatile signed __int32 *v0; // rbx
  __int64 result; // rax

  v0 = (volatile signed __int32 *)&unk_4FA10B0;
  do
  {
    result = (unsigned int)_InterlockedCompareExchange(v0, 3, 2);
    if ( (_DWORD)result == 2 )
    {
      (*((void (__fastcall **)(_QWORD))v0 - 2))(*((_QWORD *)v0 - 1));
      *((_QWORD *)v0 - 2) = 0;
      *((_QWORD *)v0 - 1) = 0;
      result = (unsigned int)_InterlockedExchange(v0, 0);
    }
    v0 += 6;
  }
  while ( v0 != (volatile signed __int32 *)&qword_4FA1170 );
  return result;
}
