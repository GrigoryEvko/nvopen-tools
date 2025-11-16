// Function: sub_2F65770
// Address: 0x2f65770
//
__int64 __fastcall sub_2F65770(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 < 0 )
    result = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    result = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  for ( ; result && (*(_BYTE *)(result + 3) & 0x10) != 0; result = *(_QWORD *)(result + 32) )
    ;
  return result;
}
