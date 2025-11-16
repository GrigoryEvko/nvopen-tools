// Function: sub_39A67A0
// Address: 0x39a67a0
//
void __fastcall sub_39A67A0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 *i; // rbx
  __int64 v6; // r15
  __int64 v7; // rax

  if ( a3 )
  {
    v4 = 8LL * *(unsigned int *)(a3 + 8);
    for ( i = (__int64 *)(a3 - v4); (__int64 *)a3 != i; ++i )
    {
      v6 = *i;
      v7 = sub_39A5A90((__int64)a1, 49, a2, 0);
      sub_39A6760(a1, v7, v6, 73);
    }
  }
}
