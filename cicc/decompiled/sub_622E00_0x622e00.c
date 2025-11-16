// Function: sub_622E00
// Address: 0x622e00
//
__int64 __fastcall sub_622E00(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rax

  v1 = *a1;
  if ( !*a1 || (unsigned __int8)(*(_BYTE *)(v1 + 80) - 10) > 1u || (*((_BYTE *)a1 + 122) & 1) == 0 )
    return sub_684B00(3117, a1 + 6);
  for ( i = *(_QWORD *)(*(_QWORD *)(v1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(i + 168);
  if ( (*(_BYTE *)(result + 16) & 2) == 0 )
    return sub_684B00(3117, a1 + 6);
  return result;
}
