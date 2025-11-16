// Function: sub_2B19330
// Address: 0x2b19330
//
char __fastcall sub_2B19330(__int64 *a1, int a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 *v11; // r13
  char result; // al
  int v13; // [rsp+8h] [rbp-48h] BYREF
  int v14; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v15[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *a1;
  v5 = a1[1];
  v13 = a2;
  v14 = a3;
  v6 = *(unsigned int *)(v4 + 8);
  v7 = *(__int64 **)v4;
  v15[2] = v5;
  v15[0] = &v14;
  v6 *= 8;
  v15[1] = &v13;
  v8 = (__int64 *)((char *)v7 + v6);
  v9 = v6 >> 3;
  v10 = v6 >> 5;
  if ( v10 )
  {
    v11 = &v7[4 * v10];
    while ( (unsigned __int8)sub_2B19110((__int64)v15, *v7) )
    {
      if ( !(unsigned __int8)sub_2B19110((__int64)v15, v7[1]) )
        return v8 == v7 + 1;
      if ( !(unsigned __int8)sub_2B19110((__int64)v15, v7[2]) )
        return v8 == v7 + 2;
      if ( !(unsigned __int8)sub_2B19110((__int64)v15, v7[3]) )
        return v8 == v7 + 3;
      v7 += 4;
      if ( v7 == v11 )
      {
        v9 = v8 - v7;
        goto LABEL_11;
      }
    }
    return v8 == v7;
  }
LABEL_11:
  if ( v9 == 2 )
    goto LABEL_17;
  if ( v9 == 3 )
  {
    if ( !(unsigned __int8)sub_2B19110((__int64)v15, *v7) )
      return v8 == v7;
    ++v7;
LABEL_17:
    if ( (unsigned __int8)sub_2B19110((__int64)v15, *v7) )
    {
      ++v7;
      goto LABEL_19;
    }
    return v8 == v7;
  }
  if ( v9 != 1 )
    return 1;
LABEL_19:
  result = sub_2B19110((__int64)v15, *v7);
  if ( !result )
    return v8 == v7;
  return result;
}
