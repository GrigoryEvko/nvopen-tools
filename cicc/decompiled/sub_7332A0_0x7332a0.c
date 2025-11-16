// Function: sub_7332A0
// Address: 0x7332a0
//
__int64 __fastcall sub_7332A0(__int64 a1, int a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rdx
  __int64 result; // rax

  v3 = qword_4F04C68[0] + 776LL * a2;
  v4 = sub_732EF0(v3);
  result = *(_QWORD *)(v3 + 24);
  if ( !result )
    result = v3 + 32;
  if ( *((_QWORD *)v4 + 23) )
    **(_QWORD **)(result + 88) = a1;
  else
    *((_QWORD *)v4 + 23) = a1;
  *(_QWORD *)(result + 88) = a1;
  return result;
}
