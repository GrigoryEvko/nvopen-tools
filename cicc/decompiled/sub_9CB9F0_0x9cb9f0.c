// Function: sub_9CB9F0
// Address: 0x9cb9f0
//
__int64 __fastcall sub_9CB9F0(__int64 a1, unsigned int *a2)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdx
  __int64 result; // rax
  _BYTE *v6; // rsi
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_QWORD **)a1;
  v3 = *a2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 528LL);
  if ( v3 >= (*(_QWORD *)(*(_QWORD *)a1 + 536LL) - v4) >> 3 )
    return 0;
  result = *(_QWORD *)(v4 + 8 * v3);
  if ( !result )
  {
    result = sub_BCC900(v2[54], a2, v4);
    v6 = (_BYTE *)v2[252];
    v7[0] = result;
    if ( v6 == (_BYTE *)v2[253] )
    {
      sub_9CABF0((__int64)(v2 + 251), v6, v7);
      result = v7[0];
    }
    else
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = result;
        v6 = (_BYTE *)v2[252];
      }
      v2[252] = v6 + 8;
    }
    *(_QWORD *)(v2[66] + 8 * v3) = result;
  }
  return result;
}
