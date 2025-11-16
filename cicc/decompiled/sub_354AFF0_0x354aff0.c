// Function: sub_354AFF0
// Address: 0x354aff0
//
unsigned __int64 __fastcall sub_354AFF0(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 *v3; // rdx
  unsigned __int64 result; // rax
  __int64 v5; // rdx

  v2 = a1[5];
  if ( ((((__int64)(a1[9] - v2) >> 3) - 1) << 6) + ((__int64)(a1[6] - a1[7]) >> 3) + ((__int64)(a1[4] - a1[2]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( v2 == *a1 )
  {
    sub_354AE70(a1, 1u, 1);
    v2 = a1[5];
  }
  *(_QWORD *)(v2 - 8) = sub_22077B0(0x200u);
  v3 = (unsigned __int64 *)(a1[5] - 8);
  a1[5] = (unsigned __int64)v3;
  result = *v3;
  v5 = *v3 + 512;
  a1[3] = result;
  a1[4] = v5;
  a1[2] = result + 504;
  *(_QWORD *)(result + 504) = *a2;
  return result;
}
