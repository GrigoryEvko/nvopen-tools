// Function: sub_D597A0
// Address: 0xd597a0
//
__int64 __fastcall sub_D597A0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 result; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 *v11; // r14

  v3 = a1[9];
  v4 = a1[6];
  v5 = a1[5];
  result = ((((v3 - v5) >> 3) - 1) << 6) + ((v4 - a1[7]) >> 3) + ((a1[4] - a1[2]) >> 3);
  if ( 0xFFFFFFFFFFFFFFFLL - result < a2 )
    sub_4262D8((__int64)"deque::_M_new_elements_at_front");
  v7 = (a2 + 63) >> 6;
  if ( v7 > (v5 - *a1) >> 3 )
    result = sub_D58C10(a1, (a2 + 63) >> 6, 1);
  if ( v7 )
  {
    v8 = -8;
    v9 = 8 * ~v7;
    do
    {
      v10 = a1[5];
      result = sub_22077B0(512);
      v11 = (__int64 *)(v8 + v10);
      v8 -= 8;
      *v11 = result;
    }
    while ( v9 != v8 );
  }
  return result;
}
