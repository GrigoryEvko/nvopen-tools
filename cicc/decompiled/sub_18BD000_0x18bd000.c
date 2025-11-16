// Function: sub_18BD000
// Address: 0x18bd000
//
char **__fastcall sub_18BD000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  char **result; // rax
  bool v13; // zf
  char **v14; // r12
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v16[8]; // [rsp+10h] [rbp-40h] BYREF

  result = (char **)&v15;
  v13 = *(_BYTE *)(a2 + 25) == 0;
  v15 = a3;
  v16[0] = a4;
  v16[1] = a1;
  v16[2] = &v15;
  if ( !v13 || (result = *(char ***)(a2 + 32), *(char ***)(a2 + 40) != result) )
    *a4 = 1;
  if ( !*(_BYTE *)(a2 + 24) )
    result = sub_18BC140((__int64)v16, (char ***)a2, a5, a6, a7, a8, a9, a10, a11, a12);
  v14 = *(char ***)(a2 + 80);
  if ( v14 != (char **)(a2 + 64) )
  {
    do
    {
      if ( *((_BYTE *)v14 + 81) || v14[12] != v14[11] )
        *(_BYTE *)v16[0] = 1;
      if ( !*((_BYTE *)v14 + 80) )
        sub_18BC140((__int64)v16, (char ***)v14 + 7, a5, a6, a7, a8, a9, a10, a11, a12);
      result = (char **)sub_220EEE0(v14);
      v14 = result;
    }
    while ( (char **)(a2 + 64) != result );
  }
  return result;
}
