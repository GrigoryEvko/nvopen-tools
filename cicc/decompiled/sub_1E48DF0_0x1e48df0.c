// Function: sub_1E48DF0
// Address: 0x1e48df0
//
__int64 __fastcall sub_1E48DF0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdx

  v2 = a1[5];
  if ( ((((a1[9] - v2) >> 3) - 1) << 6) + ((a1[6] - a1[7]) >> 3) + ((a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( v2 == *a1 )
  {
    sub_1E48C70(a1, 1u, 1);
    v2 = a1[5];
  }
  *(_QWORD *)(v2 - 8) = sub_22077B0(512);
  v3 = (__int64 *)(a1[5] - 8);
  a1[5] = (__int64)v3;
  result = *v3;
  v5 = *v3 + 512;
  a1[3] = result;
  a1[4] = v5;
  a1[2] = result + 504;
  *(_QWORD *)(result + 504) = *a2;
  return result;
}
