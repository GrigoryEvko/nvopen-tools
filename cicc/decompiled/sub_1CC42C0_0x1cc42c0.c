// Function: sub_1CC42C0
// Address: 0x1cc42c0
//
_QWORD *__fastcall sub_1CC42C0(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  _QWORD *result; // rax
  __int64 v4; // r13
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rdx

  v2 = (_QWORD *)a1[6];
  if ( v2 == (_QWORD *)(a1[8] - 8) )
  {
    v4 = a1[9];
    if ( (((__int64)v2 - a1[7]) >> 3) + ((((v4 - a1[5]) >> 3) - 1) << 6) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(a1[1] - ((v4 - *a1) >> 3)) <= 1 )
    {
      sub_1CC4140(a1, 1u, 0);
      v4 = a1[9];
    }
    *(_QWORD *)(v4 + 8) = sub_22077B0(512);
    v5 = (_QWORD *)a1[6];
    if ( v5 )
      *v5 = *a2;
    v6 = (_QWORD *)(a1[9] + 8);
    a1[9] = (__int64)v6;
    result = (_QWORD *)*v6;
    v7 = *v6 + 512LL;
    a1[7] = (__int64)result;
    a1[8] = v7;
    a1[6] = (__int64)result;
  }
  else
  {
    if ( v2 )
    {
      *v2 = *a2;
      v2 = (_QWORD *)a1[6];
    }
    result = v2 + 1;
    a1[6] = (__int64)result;
  }
  return result;
}
