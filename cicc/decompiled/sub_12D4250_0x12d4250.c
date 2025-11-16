// Function: sub_12D4250
// Address: 0x12d4250
//
__int64 __fastcall sub_12D4250(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r15
  unsigned int v5; // ebx
  _QWORD *v6; // rdi
  char v7; // al
  __int64 result; // rax

  v2 = a1 + 3;
  v3 = (_QWORD *)a1[4];
  if ( v3 == a1 + 3 )
    return 0;
  v5 = 0;
  do
  {
    v6 = v3 - 7;
    if ( !v3 )
      v6 = 0;
    v7 = sub_15E4F60(v6);
    v3 = (_QWORD *)v3[1];
    v5 += v7 == 0;
  }
  while ( v3 != v2 );
  if ( v5 <= 1 )
    return 0;
  result = *(unsigned __int8 *)(a2 + 4064);
  if ( !(_BYTE)result )
    return sub_12D3FC0(a1, a2);
  return result;
}
