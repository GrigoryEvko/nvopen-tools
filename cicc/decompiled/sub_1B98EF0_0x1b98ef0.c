// Function: sub_1B98EF0
// Address: 0x1b98ef0
//
__int64 __fastcall sub_1B98EF0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // ebx
  __int64 result; // rax

  if ( *(_QWORD *)(*(_QWORD *)a1 + 8LL) == *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
    return 1;
  v3 = a3;
  if ( byte_4FB7F60 && !a3 )
    v3 = 4;
  sub_1B98E30(a1, v3, v3);
  result = v3;
  if ( byte_4FB7F60 )
    return 1;
  return result;
}
