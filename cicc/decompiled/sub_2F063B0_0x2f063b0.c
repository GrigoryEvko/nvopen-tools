// Function: sub_2F063B0
// Address: 0x2f063b0
//
bool __fastcall sub_2F063B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9,
        __int64 a10)
{
  unsigned __int8 (__fastcall **v10)(__int64, __int64, _QWORD, __int64); // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 (__fastcall **v14)(__int64, __int64, _QWORD, __int64); // [rsp+0h] [rbp-40h]
  unsigned __int8 (__fastcall **v15)(__int64, __int64, _QWORD, __int64); // [rsp+8h] [rbp-38h]

  v10 = *(unsigned __int8 (__fastcall ***)(__int64, __int64, _QWORD, __int64))a1;
  v14 = *(unsigned __int8 (__fastcall ***)(__int64, __int64, _QWORD, __int64))(a1 + 8);
  v11 = ((__int64)v14 - *(_QWORD *)a1) >> 3;
  v12 = ((__int64)v14 - *(_QWORD *)a1) >> 5;
  if ( v12 > 0 )
  {
    v15 = &v10[4 * v12];
    while ( !(*v10)(a7, a8, *a9, a10) )
    {
      if ( v10[1](a7, a8, *a9, a10) )
        return v14 != v10 + 1;
      if ( v10[2](a7, a8, *a9, a10) )
        return v14 != v10 + 2;
      if ( v10[3](a7, a8, *a9, a10) )
        return v14 != v10 + 3;
      v10 += 4;
      if ( v10 == v15 )
      {
        v11 = v14 - v10;
        goto LABEL_10;
      }
    }
    return v14 != v10;
  }
LABEL_10:
  if ( v11 == 2 )
    goto LABEL_16;
  if ( v11 == 3 )
  {
    if ( (*v10)(a7, a8, *a9, a10) )
      return v14 != v10;
    ++v10;
LABEL_16:
    if ( !(*v10)(a7, a8, *a9, a10) )
    {
      ++v10;
      goto LABEL_18;
    }
    return v14 != v10;
  }
  if ( v11 != 1 )
    return 0;
LABEL_18:
  if ( (*v10)(a7, a8, *a9, a10) )
    return v14 != v10;
  return 0;
}
