// Function: sub_7A74B0
// Address: 0x7a74b0
//
__int64 __fastcall sub_7A74B0(int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_DWORD *)a2 = unk_4D03BE8;
  *(_QWORD *)(a2 + 8) = qword_4F083D0;
  result = *(_DWORD *)(v2 + 4) & 0x200FF;
  if ( (_DWORD)result != 131078 )
  {
    unk_4D03BE8 = a1;
    qword_4F083D0 = 0;
  }
  return result;
}
