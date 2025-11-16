// Function: sub_68B050
// Address: 0x68b050
//
__int64 __fastcall sub_68B050(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 result; // rax

  v4 = a1;
  sub_7296F0(a1, a2);
  result = qword_4F06BC0;
  *a3 = qword_4F06BC0;
  if ( result )
  {
    while ( 1 )
    {
      result = qword_4F04C68[0] + 776LL * v4;
      if ( *(_QWORD *)(result + 488) )
      {
        if ( *(_DWORD *)(result + 192) == unk_4F07270 )
          break;
      }
      v4 = *(_DWORD *)(result + 552);
    }
    qword_4F06BC0 = *(_QWORD *)(result + 488);
  }
  return result;
}
