// Function: sub_1D91300
// Address: 0x1d91300
//
__int64 *__fastcall sub_1D91300(__int64 *a1, int *a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // r13
  int v5; // ebx
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 *result; // rax
  unsigned __int8 *v9; // rsi
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a1[1];
  if ( v4 != a1[2] )
  {
    v5 = *a2;
    v6 = *a4;
    v7 = *a3;
    v10[0] = v6;
    if ( v6 )
    {
      result = (__int64 *)sub_1623A60((__int64)v10, v6, 2);
      if ( !v4 )
      {
        if ( v10[0] )
          result = (__int64 *)sub_161E7C0((__int64)v10, v10[0]);
        goto LABEL_6;
      }
    }
    else if ( !v4 )
    {
LABEL_6:
      a1[1] += 24;
      return result;
    }
    *(_DWORD *)v4 = v5;
    v9 = (unsigned __int8 *)v10[0];
    *(_QWORD *)(v4 + 8) = v7;
    *(_QWORD *)(v4 + 16) = v9;
    if ( v9 )
      result = (__int64 *)sub_1623210((__int64)v10, v9, v4 + 16);
    goto LABEL_6;
  }
  return sub_1D91040(a1, a1[1], a2, a3, a4);
}
