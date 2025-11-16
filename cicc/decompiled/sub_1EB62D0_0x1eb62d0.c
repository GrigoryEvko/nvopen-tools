// Function: sub_1EB62D0
// Address: 0x1eb62d0
//
__int64 __fastcall sub_1EB62D0(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // rsi
  _BYTE *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 *v10; // rdi
  __int64 *v11; // rcx
  __int64 v12; // [rsp+8h] [rbp-18h] BYREF

  v12 = a2;
  v3 = (_BYTE *)a1[59];
  if ( v3 == (_BYTE *)a1[60] )
  {
    sub_1EB5DA0((__int64)(a1 + 58), v3, &v12);
    v4 = (_BYTE *)a1[59];
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = a2;
      v3 = (_BYTE *)a1[59];
    }
    v4 = v3 + 8;
    a1[59] = v3 + 8;
  }
  v5 = a1[58];
  v6 = *((_QWORD *)v4 - 1);
  result = (__int64)&v4[-v5];
  v8 = (result >> 3) - 1;
  v9 = ((result >> 3) - 2) / 2;
  if ( v8 > 0 )
  {
    while ( 1 )
    {
      v10 = (__int64 *)(v5 + 8 * v9);
      v11 = (__int64 *)(v5 + 8 * v8);
      result = *v10;
      if ( *(float *)(v6 + 116) <= *(float *)(*v10 + 116) )
        break;
      *v11 = result;
      v8 = v9;
      result = (v9 - 1) / 2;
      if ( v9 <= 0 )
      {
        *v10 = v6;
        return result;
      }
      v9 = (v9 - 1) / 2;
    }
  }
  else
  {
    v11 = (__int64 *)(v5 + result - 8);
  }
  *v11 = v6;
  return result;
}
