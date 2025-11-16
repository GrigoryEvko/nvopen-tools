// Function: sub_5C7230
// Address: 0x5c7230
//
__int64 __fastcall sub_5C7230(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 result; // rax
  int v7; // eax

  result = qword_4CF6E20;
  if ( qword_4CF6E20 )
    qword_4CF6E20 = *(_QWORD *)qword_4CF6E20;
  else
    result = sub_823970(40);
  *(_QWORD *)result = 0;
  if ( qword_4CF6E30 )
    *(_QWORD *)qword_4CF6E28 = result;
  else
    qword_4CF6E30 = result;
  qword_4CF6E28 = result;
  *(_QWORD *)(result + 8) = a1;
  *(_QWORD *)(result + 16) = a2;
  *(_QWORD *)(result + 24) = a3;
  *(_QWORD *)(result + 32) = *a4;
  if ( a1 )
  {
    v7 = *(unsigned __int8 *)(a1 + 80);
    *(_BYTE *)(a1 + 84) |= 8u;
    result = (unsigned int)(v7 - 10);
    if ( (unsigned __int8)result <= 1u && !*(_QWORD *)(*(_QWORD *)(a1 + 88) + 256LL) )
      return sub_726210();
  }
  return result;
}
