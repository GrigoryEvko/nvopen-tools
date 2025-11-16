// Function: sub_139C100
// Address: 0x139c100
//
__int64 __fastcall sub_139C100(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 result; // rax

  v2 = sub_1648700(a2);
  if ( v2 != *(_QWORD *)(a1 + 16) )
    return (unsigned int)sub_139BE50((_QWORD *)a1, v2) ^ 1;
  result = *(unsigned __int8 *)(a1 + 33);
  if ( (_BYTE)result )
    return (unsigned int)sub_139BE50((_QWORD *)a1, v2) ^ 1;
  return result;
}
