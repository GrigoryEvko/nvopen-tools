// Function: sub_1EB61D0
// Address: 0x1eb61d0
//
__int64 __fastcall sub_1EB61D0(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // rsi
  _BYTE *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r8
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // rdi
  __int64 *v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-18h] BYREF

  v11 = a2;
  v2 = (_BYTE *)a1[88];
  if ( v2 == (_BYTE *)a1[89] )
  {
    sub_1EB5DA0((__int64)(a1 + 87), v2, &v11);
    v3 = (_BYTE *)a1[88];
  }
  else
  {
    if ( v2 )
    {
      *(_QWORD *)v2 = v11;
      v2 = (_BYTE *)a1[88];
    }
    v3 = v2 + 8;
    a1[88] = v2 + 8;
  }
  v4 = a1[87];
  v5 = *((_QWORD *)v3 - 1);
  result = (__int64)&v3[-v4];
  v7 = (result >> 3) - 1;
  v8 = ((result >> 3) - 2) / 2;
  if ( v7 > 0 )
  {
    while ( 1 )
    {
      v9 = (__int64 *)(v4 + 8 * v8);
      v10 = (__int64 *)(v4 + 8 * v7);
      result = *v9;
      if ( *(float *)(v5 + 116) <= *(float *)(*v9 + 116) )
        break;
      *v10 = result;
      v7 = v8;
      result = (v8 - 1) / 2;
      if ( v8 <= 0 )
      {
        *v9 = v5;
        return result;
      }
      v8 = (v8 - 1) / 2;
    }
  }
  else
  {
    v10 = (__int64 *)(v4 + result - 8);
  }
  *v10 = v5;
  return result;
}
