// Function: sub_5D09C0
// Address: 0x5d09c0
//
__int64 __fastcall sub_5D09C0(int a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax

  v1 = (__int64 *)qword_4CF6E40;
  if ( !qword_4CF6E40 )
    return sub_684B30(1679, &dword_4F063F8);
  if ( a1 )
  {
    if ( (*(_BYTE *)(qword_4CF6E40 + 9) & 1) == 0 )
      sub_684B30(1678, &dword_4F063F8);
  }
  qword_4CF6E40 = *v1;
  result = qword_4CF6E38;
  *v1 = qword_4CF6E38;
  qword_4CF6E38 = (__int64)v1;
  return result;
}
