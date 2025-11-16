// Function: sub_161E9C0
// Address: 0x161e9c0
//
__int64 __fastcall sub_161E9C0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rbx
  __int64 v3; // rsi

  v1 = 8LL * *(unsigned int *)(a1 + 8);
  if ( a1 != a1 - v1 )
  {
    v2 = a1;
    do
    {
      v3 = *(_QWORD *)(v2 - 8);
      v2 -= 8;
      if ( v3 )
        sub_161E7C0(v2, v3);
    }
    while ( v2 != a1 - v1 );
  }
  return j___libc_free_0(a1 - v1);
}
