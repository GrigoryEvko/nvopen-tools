// Function: sub_2B20DA0
// Address: 0x2b20da0
//
char __fastcall sub_2B20DA0(__int64 *a1, int a2, int a3)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 *v7; // r14
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 *v10; // r13
  char result; // al
  int v12; // [rsp+8h] [rbp-48h] BYREF
  int v13; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v14[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = *a1;
  v13 = a3;
  v12 = a2;
  v4 = *(__int64 **)v3;
  v5 = *(unsigned int *)(v3 + 8);
  v6 = a1[1];
  v14[3] = a1[2];
  v5 *= 8;
  v14[0] = v6;
  v7 = (__int64 *)((char *)v4 + v5);
  v14[1] = &v13;
  v14[2] = &v12;
  v8 = v5 >> 3;
  v9 = v5 >> 5;
  if ( v9 )
  {
    v10 = &v4[4 * v9];
    while ( (unsigned __int8)sub_2B20720((__int64)v14, *v4) )
    {
      if ( !(unsigned __int8)sub_2B20720((__int64)v14, v4[1]) )
        return v7 == v4 + 1;
      if ( !(unsigned __int8)sub_2B20720((__int64)v14, v4[2]) )
        return v7 == v4 + 2;
      if ( !(unsigned __int8)sub_2B20720((__int64)v14, v4[3]) )
        return v7 == v4 + 3;
      v4 += 4;
      if ( v4 == v10 )
      {
        v8 = v7 - v4;
        goto LABEL_11;
      }
    }
    return v7 == v4;
  }
LABEL_11:
  if ( v8 == 2 )
    goto LABEL_17;
  if ( v8 == 3 )
  {
    if ( !(unsigned __int8)sub_2B20720((__int64)v14, *v4) )
      return v7 == v4;
    ++v4;
LABEL_17:
    if ( (unsigned __int8)sub_2B20720((__int64)v14, *v4) )
    {
      ++v4;
      goto LABEL_19;
    }
    return v7 == v4;
  }
  if ( v8 != 1 )
    return 1;
LABEL_19:
  result = sub_2B20720((__int64)v14, *v4);
  if ( !result )
    return v7 == v4;
  return result;
}
