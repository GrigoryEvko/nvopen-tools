// Function: sub_CE8F50
// Address: 0xce8f50
//
__int64 __fastcall sub_CE8F50(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // edx
  char *v3; // rax
  char *v5; // [rsp+20h] [rbp-30h] BYREF
  int v6; // [rsp+28h] [rbp-28h]
  char v7; // [rsp+30h] [rbp-20h] BYREF

  sub_CE8D40((__int64)&v5, a1);
  if ( v6 )
  {
    v2 = 1;
    v3 = v5;
    do
    {
      v2 *= *(_DWORD *)v3;
      v3 += 4;
    }
    while ( &v5[4 * v6] != v3 );
    v1 = v2;
  }
  if ( v5 != &v7 )
    _libc_free(v5, a1);
  return v1;
}
