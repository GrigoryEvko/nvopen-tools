// Function: sub_124CD00
// Address: 0x124cd00
//
__int64 __fastcall sub_124CD00(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // rsi
  __int64 v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = a2;
  v2 = (_BYTE *)a1[13];
  if ( v2 == (_BYTE *)a1[14] )
  {
    sub_124CB70((__int64)(a1 + 12), v2, v8);
    v3 = v8[0];
  }
  else
  {
    v3 = v8[0];
    if ( v2 )
    {
      *(_QWORD *)v2 = v8[0];
      v2 = (_BYTE *)a1[13];
    }
    a1[13] = v2 + 8;
  }
  v4 = *(_QWORD **)(v3 + 128);
  v5 = *(_QWORD *)(v3 + 136);
  v6 = sub_C94890(v4, v5);
  sub_C0CA60((__int64)(a1 + 4), (__int64)v4, (v6 << 32) | (unsigned int)v5);
  return (__int64)(a1[13] - a1[12]) >> 3;
}
