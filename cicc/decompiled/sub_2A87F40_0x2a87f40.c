// Function: sub_2A87F40
// Address: 0x2a87f40
//
__int64 __fastcall sub_2A87F40(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 *v4; // r15
  __int64 *v5; // rbx
  unsigned int v6; // r12d
  __int64 v7; // rdx
  __int64 *v9; // [rsp+0h] [rbp-60h] BYREF
  int v10; // [rsp+8h] [rbp-58h]
  char v11; // [rsp+10h] [rbp-50h] BYREF

  sub_D47CF0(&v9, a1);
  v3 = v9;
  v4 = &v9[v10];
  if ( v4 == v9 )
  {
    v6 = 0;
  }
  else
  {
    v5 = v9;
    v6 = 0;
    do
    {
      v7 = *v5++;
      v6 |= sub_2A86DC0(a2, a1, v7);
    }
    while ( v4 != v5 );
    v3 = v9;
  }
  if ( v3 != (__int64 *)&v11 )
    _libc_free((unsigned __int64)v3);
  return v6;
}
