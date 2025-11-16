// Function: sub_8602F0
// Address: 0x8602f0
//
__int64 sub_8602F0()
{
  __int64 v0; // rdx
  __int64 result; // rax

  v0 = qword_4F04C68[0];
  result = (int)dword_4F04C60;
  do
  {
    result = v0 + 776 * result;
    if ( !result )
      break;
    *(_BYTE *)(result + 11) &= ~0x10u;
    result = *(int *)(result + 552);
  }
  while ( (_DWORD)result != -1 );
  return result;
}
