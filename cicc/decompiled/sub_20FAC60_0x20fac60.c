// Function: sub_20FAC60
// Address: 0x20fac60
//
_QWORD *__fastcall sub_20FAC60(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *result; // rax
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rdx
  _QWORD *v7; // [rsp+0h] [rbp-70h]
  unsigned __int8 *v8; // [rsp+8h] [rbp-68h] BYREF
  char v9; // [rsp+17h] [rbp-59h] BYREF
  __int64 v10; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int8 **v11; // [rsp+20h] [rbp-50h] BYREF
  __int64 v12; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v13[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = (unsigned __int8 *)a2;
  v8 = (unsigned __int8 *)sub_15B1030(a2);
  v2 = sub_20FABF0((_QWORD *)(a1 + 120), (__int64 *)&v8);
  if ( v2 )
    return v2 + 2;
  v10 = 0;
  if ( (unsigned int)*v8 - 18 <= 1 )
    v10 = sub_20FAC60(a1, *(_QWORD *)&v8[8 * (1LL - *((unsigned int *)v8 + 2))]);
  v9 = 1;
  v13[0] = &v9;
  v13[1] = &v12;
  v12 = 0;
  v13[2] = &v8;
  v13[3] = &v10;
  v11 = &v8;
  result = sub_20FA530((_QWORD *)(a1 + 120), (unsigned __int64 **)&v11, (__int64)v13) + 2;
  if ( *v8 == 17 )
  {
    v6 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 188) )
    {
      v7 = result;
      sub_16CD150(a1 + 176, (const void *)(a1 + 192), 0, 8, v4, v5);
      v6 = *(unsigned int *)(a1 + 184);
      result = v7;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * v6) = result;
    ++*(_DWORD *)(a1 + 184);
  }
  return result;
}
