// Function: sub_709290
// Address: 0x709290
//
__int64 __fastcall sub_709290(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 result; // rax

  if ( !dword_4D04944 && qword_4D0495C )
    sub_603790(a1, a2);
  sub_825150();
  sub_863FC0(a1, a2, v2, v3, v4);
  sub_860100(1);
  if ( !dword_4D04944 )
    sub_8CFCF0();
  sub_723F40(0);
  sub_8628A0();
  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    result = (unsigned int)(dword_4D04944 | unk_4D03FE8);
    if ( !(dword_4D04944 | unk_4D03FE8) )
      return sub_89A020();
  }
  return result;
}
