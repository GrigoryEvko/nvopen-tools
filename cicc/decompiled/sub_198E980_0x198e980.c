// Function: sub_198E980
// Address: 0x198e980
//
__int64 *__fastcall sub_198E980(__int64 *a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // r12
  __int64 *v12; // [rsp+10h] [rbp-40h]
  unsigned __int64 v13; // [rsp+18h] [rbp-38h]

  v4 = a2 - (_QWORD)a1;
  v5 = v4 >> 3;
  if ( v4 <= 0 )
    return a1;
  v12 = a1;
  do
  {
    while ( 1 )
    {
      v7 = v5 >> 1;
      v8 = &v12[v5 >> 1];
      v9 = *a3;
      v13 = sub_1368AA0(a4, *v8);
      if ( v13 > sub_1368AA0(a4, v9) )
        break;
      v5 = v5 - v7 - 1;
      v12 = v8 + 1;
      if ( v5 <= 0 )
        return v12;
    }
    v5 >>= 1;
  }
  while ( v7 > 0 );
  return v12;
}
