// Function: sub_6FC450
// Address: 0x6fc450
//
__int64 __fastcall sub_6FC450(__m128i *a1, unsigned __int8 a2)
{
  __int64 i; // rbx
  int v4; // r8d
  __int64 result; // rax
  __int64 v6; // rdi

  for ( i = a1->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = sub_8D2A90(i);
  result = 14;
  if ( v4 )
    result = *(unsigned __int8 *)(i + 160);
  if ( (_BYTE)result != a2 )
  {
    if ( *(_BYTE *)(i + 140) == 4 )
      v6 = sub_72C7D0(a2);
    else
      v6 = sub_72C610(a2);
    return sub_6FC3F0(v6, a1, 1u);
  }
  return result;
}
