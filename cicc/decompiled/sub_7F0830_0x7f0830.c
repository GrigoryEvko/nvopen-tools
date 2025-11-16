// Function: sub_7F0830
// Address: 0x7f0830
//
void *__fastcall sub_7F0830(_BYTE *a1)
{
  void *v1; // r12
  __int64 *v2; // r12
  _QWORD *v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9

  v1 = a1;
  if ( unk_4F07520 && (a1[24] != 1 || !sub_730FB0(a1[56]))
    || (unsigned int)sub_7E1F90(*(_QWORD *)a1)
    || (unsigned int)sub_7E1F40(*(_QWORD *)a1) )
  {
    v2 = sub_7E1E70((__int64 *)a1);
    v3 = sub_72BA30(5u);
    v1 = sub_73DBF0(0x3Bu, (__int64)v3, (__int64)v2);
    sub_7F07E0((__int64)v1, (__int64)v3, v4, v5, v6, v7);
  }
  else
  {
    a1[26] |= 0x10u;
  }
  return v1;
}
