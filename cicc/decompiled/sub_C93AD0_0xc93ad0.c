// Function: sub_C93AD0
// Address: 0xc93ad0
//
__int64 __fastcall sub_C93AD0(__int64 *a1, _WORD *a2, size_t a3)
{
  __int64 v3; // r12
  unsigned __int64 i; // rcx
  __int64 v6; // rax

  v3 = 0;
  if ( a3 )
  {
    for ( i = 0; ; i = a3 + v6 )
    {
      v6 = sub_C931B0(a1, a2, a3, i);
      if ( v6 == -1 )
        break;
      ++v3;
    }
  }
  return v3;
}
