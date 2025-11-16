// Function: sub_3440F60
// Address: 0x3440f60
//
char __fastcall sub_3440F60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD **a5,
        unsigned __int8 a6,
        unsigned int a7)
{
  __int64 (*v7)(); // rax
  char result; // al
  char v11; // r8
  __int64 *v12; // rbx
  signed __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 *v15; // r14
  unsigned int v16; // r14d
  __int64 *v17; // [rsp+8h] [rbp-38h]

  v7 = *(__int64 (**)())(*(_QWORD *)a1 + 2088LL);
  if ( v7 == sub_343F030 )
    return 0;
  v11 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD **, _QWORD, __int64, _QWORD))v7)(
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          1,
          a7);
  result = 0;
  if ( v11 )
    return result;
  v12 = *(__int64 **)(a2 + 40);
  v13 = *(unsigned int *)(a2 + 64);
  v17 = &v12[5 * v13];
  if ( v13 >> 2 )
  {
    v14 = a7 + 1;
    v15 = &v12[20 * (v13 >> 2)];
    while ( (unsigned __int8)sub_33DE850(a5, *v12, v12[1], a6, v14) )
    {
      if ( !(unsigned __int8)sub_33DE850(a5, v12[5], v12[6], a6, v14) )
        return v17 == v12 + 5;
      if ( !(unsigned __int8)sub_33DE850(a5, v12[10], v12[11], a6, v14) )
        return v17 == v12 + 10;
      if ( !(unsigned __int8)sub_33DE850(a5, v12[15], v12[16], a6, v14) )
        return v17 == v12 + 15;
      v12 += 20;
      if ( v15 == v12 )
      {
        v13 = 0xCCCCCCCCCCCCCCCDLL * (v17 - v12);
        goto LABEL_14;
      }
    }
    return v12 == v17;
  }
LABEL_14:
  if ( v13 == 2 )
  {
    v16 = a7 + 1;
LABEL_23:
    if ( !(unsigned __int8)sub_33DE850(a5, *v12, v12[1], a6, v16) )
      return v17 == v12;
    v12 += 5;
    goto LABEL_25;
  }
  if ( v13 == 3 )
  {
    v16 = a7 + 1;
    if ( !(unsigned __int8)sub_33DE850(a5, *v12, v12[1], a6, a7 + 1) )
      return v12 == v17;
    v12 += 5;
    goto LABEL_23;
  }
  if ( v13 != 1 )
    return 1;
  v16 = a7 + 1;
LABEL_25:
  result = sub_33DE850(a5, *v12, v12[1], a6, v16);
  if ( !result )
    return v17 == v12;
  return result;
}
