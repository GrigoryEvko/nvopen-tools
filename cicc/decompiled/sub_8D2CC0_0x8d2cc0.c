// Function: sub_8D2CC0
// Address: 0x8d2cc0
//
_BOOL8 __fastcall sub_8D2CC0(__int64 a1)
{
  __int64 i; // rbx
  _BOOL8 result; // rax
  __int64 v3; // rdx

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = sub_8D2820(i);
  if ( result )
  {
    v3 = *(_QWORD *)(i + 128);
    if ( (unsigned __int64)(v3 - 1) <= 1 )
      return (qword_4F06A7C == 0) == byte_4B6DF90[*(unsigned __int8 *)(i + 160)];
    result = 0;
    if ( v3 == 8 )
      return (qword_4F06A7C == 0) == byte_4B6DF90[*(unsigned __int8 *)(i + 160)];
  }
  return result;
}
