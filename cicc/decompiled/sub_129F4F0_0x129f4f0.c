// Function: sub_129F4F0
// Address: 0x129f4f0
//
__int64 __fastcall sub_129F4F0(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // r13
  __int64 *v9; // rdi
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // rdx

  v4 = a1[8];
  v5 = (_QWORD *)a1[6];
  result = v4 - 8;
  if ( v5 == (_QWORD *)result )
  {
    v8 = a1[9];
    if ( (((__int64)v5 - a1[7]) >> 3) + ((((v8 - a1[5]) >> 3) - 1) << 6) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(a1[1] - ((v8 - *a1) >> 3)) <= 1 )
    {
      sub_129F230(a1, 1u, 0);
      v8 = a1[9];
    }
    *(_QWORD *)(v8 + 8) = sub_22077B0(512);
    v9 = (__int64 *)a1[6];
    if ( v9 )
    {
      v10 = *a2;
      *v9 = *a2;
      if ( v10 )
        sub_1623A60(v9, v10, 2);
    }
    v11 = (__int64 *)(a1[9] + 8);
    a1[9] = (__int64)v11;
    result = *v11;
    v12 = *v11 + 512;
    a1[7] = result;
    a1[8] = v12;
    a1[6] = result;
  }
  else
  {
    if ( v5 )
    {
      v7 = *a2;
      *v5 = v7;
      if ( v7 )
        result = sub_1623A60(v5, v7, 2);
      v5 = (_QWORD *)a1[6];
    }
    a1[6] = (__int64)(v5 + 1);
  }
  return result;
}
