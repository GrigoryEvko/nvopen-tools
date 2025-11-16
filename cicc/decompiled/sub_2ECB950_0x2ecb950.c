// Function: sub_2ECB950
// Address: 0x2ecb950
//
unsigned __int64 *__fastcall sub_2ECB950(_QWORD *a1)
{
  unsigned __int64 *result; // rax
  __int64 v3; // rdi
  __int64 (*v4)(); // rax

  if ( qword_5020E28 != sub_2EC0BB0 )
    return (unsigned __int64 *)qword_5020E28();
  v3 = a1[4];
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 40LL);
  if ( v4 == sub_23CE2A0 )
    return sub_2ECB630(a1);
  result = (unsigned __int64 *)((__int64 (__fastcall *)(__int64, _QWORD *))v4)(v3, a1);
  if ( !result )
    return sub_2ECB630(a1);
  return result;
}
