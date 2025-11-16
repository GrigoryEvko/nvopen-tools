// Function: sub_30FB990
// Address: 0x30fb990
//
unsigned __int8 *__fastcall sub_30FB990(unsigned __int8 *a1)
{
  unsigned __int8 *result; // rax
  __int64 v2; // rdx
  __int64 v4; // rdi
  bool v5; // r8

  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return 0;
  v2 = 0x8000000000041LL;
  if ( !_bittest64(&v2, (unsigned int)*a1 - 34) )
    return 0;
  v4 = *((_QWORD *)a1 - 4);
  if ( !v4 )
    return 0;
  if ( *(_BYTE *)v4 )
    return 0;
  if ( *(_QWORD *)(v4 + 24) != *((_QWORD *)a1 + 10) )
    return 0;
  v5 = sub_B2FC80(v4);
  result = a1;
  if ( v5 )
    return 0;
  return result;
}
