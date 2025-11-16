// Function: sub_5E9580
// Address: 0x5e9580
//
__int64 __fastcall sub_5E9580(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v2; // rdx

  result = unk_4F04C68 + 776LL * unk_4F04C64;
  if ( *(_BYTE *)(result + 4) != 8 )
    goto LABEL_5;
  do
    result -= 776;
  while ( *(_BYTE *)(result + 4) == 8 );
  while ( *(_BYTE *)(result - 772) == 6 )
  {
    result -= 776;
LABEL_5:
    ;
  }
  v2 = *(_QWORD **)(result + 264);
  if ( v2 )
    *v2 = a1;
  else
    *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(result + 208) + 96LL) + 56LL) = a1;
  *(_QWORD *)(result + 264) = a1;
  return result;
}
