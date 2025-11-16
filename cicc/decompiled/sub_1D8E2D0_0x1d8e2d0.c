// Function: sub_1D8E2D0
// Address: 0x1d8e2d0
//
unsigned __int64 __fastcall sub_1D8E2D0(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rsi
  unsigned __int64 result; // rax
  __int64 v6; // rdi

  v2 = a1[7];
  v3 = a1[6];
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 16);
      if ( v4 )
        result = sub_161E7C0(v3 + 16, v4);
      v3 += 24;
    }
    while ( v2 != v3 );
    v3 = a1[6];
  }
  if ( v3 )
    result = j_j___libc_free_0(v3, a1[8] - v3);
  v6 = a1[3];
  if ( v6 )
    return j_j___libc_free_0(v6, a1[5] - v6);
  return result;
}
