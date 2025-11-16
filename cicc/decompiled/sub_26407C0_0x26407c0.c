// Function: sub_26407C0
// Address: 0x26407c0
//
unsigned __int64 __fastcall sub_26407C0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  unsigned __int64 result; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  unsigned __int64 v8; // r13
  unsigned __int64 *v9; // r13

  v3 = a1[9];
  if ( 0xFFFFFFFFFFFFFFFLL
     - (((((__int64)(v3 - a1[5]) >> 3) - 1) << 6)
      + ((__int64)(a1[6] - a1[7]) >> 3)
      + ((__int64)(a1[4] - a1[2]) >> 3)) < a2 )
    sub_4262D8((__int64)"deque::_M_new_elements_at_back");
  v4 = (a2 + 63) >> 6;
  result = a1[1] - ((__int64)(v3 - *a1) >> 3);
  if ( v4 + 1 > result )
    result = sub_263FD80(a1, (a2 + 63) >> 6, 0);
  if ( v4 )
  {
    v6 = 8 * (v4 + 1);
    v7 = 8;
    do
    {
      v8 = a1[9];
      result = sub_22077B0(0x200u);
      v9 = (unsigned __int64 *)(v7 + v8);
      v7 += 8;
      *v9 = result;
    }
    while ( v6 != v7 );
  }
  return result;
}
