// Function: sub_1688B40
// Address: 0x1688b40
//
_QWORD *__fastcall sub_1688B40(unsigned __int64 a1, char a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // r12

  if ( a1 > 0xFFFFFFFFFFFFFFF7LL )
    return 0;
  v2 = a1 + 8;
  v3 = (_QWORD *)malloc(a1 + 8);
  if ( !v3 )
    return 0;
  if ( !a2 )
    sub_1688B00(v2);
  *v3 = v2;
  return v3 + 1;
}
