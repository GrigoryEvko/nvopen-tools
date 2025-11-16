// Function: sub_72F950
// Address: 0x72f950
//
__int64 __fastcall sub_72F950(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(_QWORD *)(qword_4F04C68[0] + 24LL);
  v2 = *(_QWORD *)(qword_4F04C68[0] + 184LL);
  if ( !result )
    result = qword_4F04C68[0] + 32LL;
  if ( *(_QWORD *)(v2 + 192) )
    **(_QWORD **)(result + 64) = a1;
  else
    *(_QWORD *)(v2 + 192) = a1;
  *(_QWORD *)(result + 64) = a1;
  *a1 = 0;
  return result;
}
