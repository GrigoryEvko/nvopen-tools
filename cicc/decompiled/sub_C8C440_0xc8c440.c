// Function: sub_C8C440
// Address: 0xc8c440
//
__int64 sub_C8C440()
{
  volatile signed __int32 *v0; // rbx
  __int64 result; // rax

  if ( !byte_4F84BE0 && (unsigned int)sub_2207590(&byte_4F84BE0) )
    sub_2207640(&byte_4F84BE0);
  v0 = (volatile signed __int32 *)&unk_4F84C10;
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
  while ( v0 != (volatile signed __int32 *)&algn_4F84CC8[8] );
  return result;
}
