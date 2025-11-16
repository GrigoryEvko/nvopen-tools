// Function: sub_2260560
// Address: 0x2260560
//
__int64 __fastcall sub_2260560(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // r15
  unsigned int v5; // ebx
  __int64 v6; // rdi
  bool v7; // al
  __int64 result; // rax

  v2 = a1 + 3;
  v3 = (_QWORD *)a1[4];
  if ( v3 == a1 + 3 )
    return 0;
  v5 = 0;
  do
  {
    v6 = (__int64)(v3 - 7);
    if ( !v3 )
      v6 = 0;
    v7 = sub_B2FC80(v6);
    v3 = (_QWORD *)v3[1];
    v5 += !v7;
  }
  while ( v3 != v2 );
  if ( v5 <= 1 )
    return 0;
  result = *(unsigned __int8 *)(a2 + 1520);
  if ( !(_BYTE)result )
    return sub_2260240(a1, a2);
  return result;
}
