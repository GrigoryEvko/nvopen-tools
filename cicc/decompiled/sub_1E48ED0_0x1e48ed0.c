// Function: sub_1E48ED0
// Address: 0x1e48ed0
//
__int64 __fastcall sub_1E48ED0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // r13
  _QWORD *v3; // rax
  __int64 *v4; // rdx
  __int64 result; // rax
  __int64 v6; // rdx

  v2 = a1[9];
  if ( ((((v2 - a1[5]) >> 3) - 1) << 6) + ((a1[6] - a1[7]) >> 3) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( (unsigned __int64)(a1[1] - ((v2 - *a1) >> 3)) <= 1 )
  {
    sub_1E48C70(a1, 1u, 0);
    v2 = a1[9];
  }
  *(_QWORD *)(v2 + 8) = sub_22077B0(512);
  v3 = (_QWORD *)a1[6];
  if ( v3 )
    *v3 = *a2;
  v4 = (__int64 *)(a1[9] + 8);
  a1[9] = (__int64)v4;
  result = *v4;
  v6 = *v4 + 512;
  a1[7] = result;
  a1[8] = v6;
  a1[6] = result;
  return result;
}
