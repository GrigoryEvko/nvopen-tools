// Function: sub_866340
// Address: 0x866340
//
__int64 sub_866340()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = unk_4F04C48;
  if ( unk_4F04C48 != -1 )
  {
    v1 = qword_4F04C68[0];
    do
    {
      result = v1 + 776 * result;
      if ( !result )
        break;
      if ( *(_BYTE *)(result + 4) == 9 )
        *(_QWORD *)(result + 656) = *(_QWORD *)(result + 648);
      result = *(int *)(result + 552);
    }
    while ( (_DWORD)result != -1 );
    ++dword_4F5FCD8;
  }
  return result;
}
