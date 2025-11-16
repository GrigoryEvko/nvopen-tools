// Function: sub_9CAD80
// Address: 0x9cad80
//
__int64 __fastcall sub_9CAD80(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 result; // rax
  _BYTE *v5; // rsi
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = (unsigned int)a2;
  v3 = a1[66];
  if ( (unsigned int)a2 >= (unsigned __int64)((a1[67] - v3) >> 3) )
    return 0;
  result = *(_QWORD *)(v3 + 8LL * (unsigned int)a2);
  if ( !result )
  {
    result = sub_BCC900(a1[54], a2, v3);
    v5 = (_BYTE *)a1[252];
    v6[0] = result;
    if ( v5 == (_BYTE *)a1[253] )
    {
      sub_9CABF0((__int64)(a1 + 251), v5, v6);
      result = v6[0];
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = result;
        v5 = (_BYTE *)a1[252];
      }
      a1[252] = v5 + 8;
    }
    *(_QWORD *)(a1[66] + 8 * v2) = result;
  }
  return result;
}
