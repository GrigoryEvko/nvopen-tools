// Function: sub_809040
// Address: 0x809040
//
unsigned __int64 __fastcall sub_809040(unsigned __int64 a1, _QWORD *a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // r13
  _QWORD *v4; // rdi
  unsigned __int64 i; // rbx
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // r12

  v2 = 1;
  v3 = a1;
  do
    v2 *= 36LL;
  while ( a1 >= v2 );
  v4 = (_QWORD *)qword_4F18BE0;
  for ( i = v2 / 0x24; ; i /= 0x24u )
  {
    ++*a2;
    v7 = v4[2];
    v8 = v3 / i;
    if ( (unsigned __int64)(v7 + 1) > v4[1] )
    {
      sub_823810(v4);
      v4 = (_QWORD *)qword_4F18BE0;
      v7 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v4[4] + v7) = a0123456789abcd_3[(unsigned int)v8];
    v3 -= i * (unsigned int)v8;
    ++v4[2];
    result = 0xE38E38E38E38E38FLL * i;
    if ( i <= 0x23 )
      break;
  }
  return result;
}
