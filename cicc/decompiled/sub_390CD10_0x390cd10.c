// Function: sub_390CD10
// Address: 0x390cd10
//
__int64 __fastcall sub_390CD10(__int64 *a1, __int64 **a2)
{
  __int64 *v2; // r15
  unsigned int v3; // r12d
  __int64 v4; // rbx
  unsigned int v5; // eax
  __int64 *v7; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)a1[4];
  v7 = (__int64 *)a1[5];
  if ( v2 == v7 )
  {
    return 0;
  }
  else
  {
    v3 = 0;
    do
    {
      v4 = *v2;
      v5 = v3;
      do
      {
        v3 = v5;
        v5 = sub_390CBA0(a1, a2, v4);
      }
      while ( (_BYTE)v5 );
      ++v2;
    }
    while ( v7 != v2 );
  }
  return v3;
}
