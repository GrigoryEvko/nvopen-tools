// Function: sub_72C470
// Address: 0x72c470
//
__int64 __fastcall sub_72C470(int a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  __int64 result; // rax

  if ( dword_4F077C4 == 2 )
    v2 = unk_4F06B39;
  else
    v2 = unk_4F06B38;
  sub_72BAF0(a2, a1, v2);
  result = sub_72C390();
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
