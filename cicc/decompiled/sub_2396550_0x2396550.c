// Function: sub_2396550
// Address: 0x2396550
//
void __fastcall sub_2396550(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 *v2; // rax
  unsigned __int64 v3; // r12

  v1 = *a1;
  if ( *a1 )
  {
    if ( (v1 & 4) != 0 )
    {
      v2 = (unsigned __int64 *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      v3 = (unsigned __int64)v2;
      if ( v2 )
      {
        if ( (unsigned __int64 *)*v2 != v2 + 2 )
          _libc_free(*v2);
        j_j___libc_free_0(v3);
      }
    }
  }
}
