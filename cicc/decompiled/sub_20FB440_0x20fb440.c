// Function: sub_20FB440
// Address: 0x20fb440
//
_QWORD *__fastcall sub_20FB440(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *result; // rax
  unsigned __int8 *v4; // [rsp+8h] [rbp-68h] BYREF
  char v5; // [rsp+17h] [rbp-59h] BYREF
  __int64 v6; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int8 **v7; // [rsp+20h] [rbp-50h] BYREF
  __int64 v8; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v9[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = (unsigned __int8 *)sub_15B1030(a2);
  v2 = sub_20FABF0(a1 + 1, (__int64 *)&v4);
  if ( v2 )
    return v2 + 2;
  v6 = 0;
  if ( (unsigned int)*v4 - 18 <= 1 )
    v6 = sub_20FB390(a1, *(unsigned __int8 **)&v4[8 * (1LL - *((unsigned int *)v4 + 2))], 0);
  v5 = 0;
  v9[0] = &v5;
  v9[1] = &v8;
  v8 = 0;
  v9[2] = &v4;
  v9[3] = &v6;
  v7 = &v4;
  result = sub_20FA530(a1 + 1, (unsigned __int64 **)&v7, (__int64)v9) + 2;
  if ( !v6 )
    a1[28] = result;
  return result;
}
